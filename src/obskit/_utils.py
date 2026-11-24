"""Fonctions utilitaires internes (validation des entrées, petits calculs)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def verifier_colonnes(df: pd.DataFrame, colonnes: Iterable[str]) -> None:
    """Lève une ``KeyError`` explicite si des colonnes sont absentes de ``df``."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` doit être un pandas.DataFrame.")
    manquantes = [c for c in colonnes if c is not None and c not in df.columns]
    if manquantes:
        raise KeyError(
            f"Colonne(s) absente(s) du DataFrame : {manquantes}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )


def en_liste(x: str | Sequence[str] | None) -> list[str]:
    """Convertit ``None`` / une chaîne / une séquence en liste de chaînes."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def explicatives_par_defaut(
    df: pd.DataFrame, cible: str | Sequence[str], explicatives: Sequence[str] | None,
    exclure: Sequence[str] | None = None,
) -> list[str]:
    """Si ``explicatives`` vaut ``None``, renvoie toutes les colonnes sauf la/les cible(s)."""
    if explicatives is not None:
        return en_liste(explicatives)
    a_exclure = set(en_liste(cible)) | set(en_liste(exclure))
    return [c for c in df.columns if c not in a_exclure]


def est_numerique(serie: pd.Series) -> bool:
    """Vrai si la série est numérique (booléens exclus)."""
    return pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)


def est_categorielle(serie: pd.Series, max_modalites_num: int = 0) -> bool:
    """Vrai si la série doit être traitée comme catégorielle.

    Les colonnes texte, ``category`` et booléennes sont catégorielles. Une colonne
    numérique est considérée catégorielle si elle a au plus ``max_modalites_num``
    valeurs distinctes (0 = jamais).
    """
    if pd.api.types.is_bool_dtype(serie) or isinstance(serie.dtype, pd.CategoricalDtype):
        return True
    if not pd.api.types.is_numeric_dtype(serie):
        return not pd.api.types.is_datetime64_any_dtype(serie)
    return max_modalites_num > 0 and serie.nunique(dropna=True) <= max_modalites_num


def poids_ou_uns(df: pd.DataFrame, poids: str | None) -> np.ndarray:
    """Renvoie le vecteur de poids (ou un vecteur de 1)."""
    if poids is None:
        return np.ones(len(df), dtype=float)
    verifier_colonnes(df, [poids])
    w = df[poids].to_numpy(dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Les poids doivent être finis et positifs ou nuls.")
    return w


def etoiles(p: float) -> str:
    """Codes de significativité usuels."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def formater_p(p: float) -> str:
    """Formate une p-valeur pour un affichage lisible (``< 0,001`` …)."""
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
