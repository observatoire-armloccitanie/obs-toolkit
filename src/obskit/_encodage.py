"""Construction des matrices de plan (variables indicatrices, interactions).

Le :class:`Encodeur` transforme un DataFrame « métier » (variables numériques
et catégorielles mélangées) en matrice numérique pour les modèles, en gardant
la mémoire des modalités de référence pour pouvoir prédire sur de nouvelles
données.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import en_liste, est_categorielle, verifier_colonnes


class Encodeur:
    """Encode des variables explicatives en matrice de plan.

    Parameters
    ----------
    explicatives : list of str
        Variables explicatives (numériques ou catégorielles).
    reference : dict, optional
        Modalité de référence par variable catégorielle, ex.
        ``{"niveau_formation": "V", "sexe": "Femme"}``.
    reference_defaut : {"frequente", "premiere"}
        Règle pour les variables sans référence explicite : modalité la plus
        fréquente (défaut, recommandé en sociologie quantitative car elle
        stabilise les estimations) ou première modalité (ordre des catégories
        ou ordre alphabétique, comme dans R).
    interactions : list of tuple, optional
        Couples de variables à croiser, ex. ``[("sexe", "qpv")]``.
    constante : bool
        Ajoute une colonne ``const`` (défaut ``True``).
    categorielles : list of str, optional
        Force le traitement catégoriel de variables numériques (codes, années…).

    Examples
    --------
    >>> enc = Encodeur(["age", "sexe"], reference={"sexe": "Femme"})
    >>> X = enc.fit_transform(df)          # doctest: +SKIP
    >>> list(X.columns)                    # doctest: +SKIP
    ['const', 'age', 'sexe[Homme]']
    """

    def __init__(
        self,
        explicatives: Sequence[str],
        reference: Mapping[str, object] | None = None,
        reference_defaut: str = "frequente",
        interactions: Sequence[tuple[str, str]] | None = None,
        constante: bool = True,
        categorielles: Sequence[str] | None = None,
    ) -> None:
        self.explicatives = en_liste(explicatives)
        self.reference = dict(reference or {})
        if reference_defaut not in ("frequente", "premiere"):
            raise ValueError("reference_defaut doit valoir 'frequente' ou 'premiere'.")
        self.reference_defaut = reference_defaut
        self.interactions = [tuple(i) for i in (interactions or [])]
        self.constante = constante
        self.categorielles_forcees = set(en_liste(categorielles))
        self.modalites_: dict[str, list] = {}
        self.references_: dict[str, object] = {}
        self.colonnes_par_variable_: dict[str, list[str]] = {}
        self.colonnes_: list[str] = []
        self._ajuste = False

    # ------------------------------------------------------------------
    def _est_cat(self, df: pd.DataFrame, var: str) -> bool:
        return var in self.categorielles_forcees or est_categorielle(df[var])

    def fit(self, df: pd.DataFrame) -> "Encodeur":
        """Apprend les modalités et les références à partir de ``df``."""
        verifier_colonnes(df, self.explicatives)
        for a, b in self.interactions:
            if a not in self.explicatives or b not in self.explicatives:
                raise ValueError(
                    f"L'interaction ({a}, {b}) porte sur des variables absentes de `explicatives`."
                )
        self.modalites_.clear()
        self.references_.clear()
        for var in self.explicatives:
            s = df[var]
            if self._est_cat(df, var):
                if isinstance(s.dtype, pd.CategoricalDtype):
                    mods = [m for m in s.cat.categories if (s == m).any()]
                else:
                    mods = sorted(s.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
                if len(mods) < 2:
                    raise ValueError(
                        f"La variable catégorielle '{var}' n'a qu'une modalité observée : "
                        "elle ne peut pas être utilisée comme explicative."
                    )
                if var in self.reference:
                    ref = self.reference[var]
                    if ref not in mods:
                        raise ValueError(
                            f"Référence '{ref}' introuvable pour '{var}'. Modalités : {mods}"
                        )
                elif self.reference_defaut == "frequente":
                    ref = s.value_counts().index[0]
                else:
                    ref = mods[0]
                self.modalites_[var] = mods
                self.references_[var] = ref
        # noms de colonnes
        self.colonnes_par_variable_ = {}
        cols: list[str] = ["const"] if self.constante else []
        for var in self.explicatives:
            if var in self.modalites_:
                vc = [f"{var}[{m}]" for m in self.modalites_[var] if m != self.references_[var]]
            else:
                vc = [var]
            self.colonnes_par_variable_[var] = vc
            cols += vc
        for a, b in self.interactions:
            vc = [f"{ca}:{cb}" for ca in self.colonnes_par_variable_[a]
                  for cb in self.colonnes_par_variable_[b]]
            self.colonnes_par_variable_[f"{a}:{b}"] = vc
            cols += vc
        self.colonnes_ = cols
        self._ajuste = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit la matrice de plan pour ``df`` (mêmes colonnes qu'à l'ajustement)."""
        if not self._ajuste:
            raise RuntimeError("Appeler `fit` avant `transform`.")
        verifier_colonnes(df, self.explicatives)
        blocs: dict[str, np.ndarray] = {}
        if self.constante:
            blocs["const"] = np.ones(len(df))
        for var in self.explicatives:
            s = df[var]
            if var in self.modalites_:
                connues = s.isin(self.modalites_[var]) | s.isna()
                if not connues.all():
                    inconnues = s[~connues].unique().tolist()[:5]
                    warnings.warn(
                        f"Modalités inconnues pour '{var}' ({inconnues}) : "
                        "codées comme la référence.", stacklevel=2,
                    )
                for m in self.modalites_[var]:
                    if m == self.references_[var]:
                        continue
                    col = (s == m).to_numpy(dtype=float, copy=True)
                    col[s.isna().to_numpy()] = np.nan
                    blocs[f"{var}[{m}]"] = col
            else:
                blocs[var] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        for a, b in self.interactions:
            for ca in self.colonnes_par_variable_[a]:
                for cb in self.colonnes_par_variable_[b]:
                    blocs[f"{ca}:{cb}"] = blocs[ca] * blocs[cb]
        return pd.DataFrame(blocs, index=df.index)[self.colonnes_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Équivalent à ``fit(df).transform(df)``."""
        return self.fit(df).transform(df)

    def description_references(self) -> pd.DataFrame:
        """Tableau des modalités de référence retenues."""
        return pd.DataFrame(
            {"variable": list(self.references_), "reference": list(self.references_.values())}
        )


def construire_matrice(
    df: pd.DataFrame,
    explicatives: Sequence[str],
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    constante: bool = True,
    reference_defaut: str = "frequente",
) -> tuple[pd.DataFrame, Encodeur]:
    """Raccourci : encode ``df[explicatives]`` et renvoie ``(X, encodeur)``.

    Voir :class:`Encodeur` pour la description des paramètres.
    """
    enc = Encodeur(explicatives, reference=reference, interactions=interactions,
                   constante=constante, reference_defaut=reference_defaut)
    return enc.fit_transform(df), enc
