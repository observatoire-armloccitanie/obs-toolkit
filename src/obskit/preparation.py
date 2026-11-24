"""Préparation et recodage des données (nettoyage, tranches, durées, regroupements)."""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import en_liste, verifier_colonnes

__all__ = [
    "nettoyer_noms_colonnes",
    "detecter_types",
    "calculer_age",
    "tranches_age",
    "discretiser",
    "duree_entre",
    "recoder",
    "regrouper_modalites_rares",
    "convertir_binaire",
    "convertir_dates",
    "imputer",
    "ordonner_modalites",
]

TRANCHES_AGE_ML = (16, 18, 22, 26)
"""Bornes usuelles des tranches d'âge Missions Locales : 16-17, 18-21, 22-25, 26 et +."""


def _slug(texte: str) -> str:
    t = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^0-9a-zA-Z]+", "_", t).strip("_").lower()
    return t or "colonne"


def nettoyer_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en ``snake_case`` sans accents ni espaces.

    Les exports OBIEE / i-milo ont des intitulés longs (« Date d'entrée en CEJ ») ;
    on obtient ``date_d_entree_en_cej``. Les doublons sont suffixés ``_2``, ``_3``…
    """
    vus: dict[str, int] = {}
    noms = []
    for c in df.columns:
        s = _slug(c)
        if s in vus:
            vus[s] += 1
            s = f"{s}_{vus[s]}"
        else:
            vus[s] = 1
        noms.append(s)
    out = df.copy()
    out.columns = noms
    return out


def detecter_types(df: pd.DataFrame, max_modalites: int = 15) -> pd.DataFrame:
    """Propose un type statistique par colonne.

    Types : ``identifiant``, ``binaire``, ``categorielle``, ``numerique``,
    ``date``, ``texte`` (trop de modalités pour une catégorielle).
    """
    lignes = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        k = s.nunique(dropna=True)
        if pd.api.types.is_datetime64_any_dtype(s):
            t = "date"
        elif k <= 2:
            t = "binaire"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if k == n and pd.api.types.is_integer_dtype(s):
                t = "identifiant"
            elif k <= max_modalites and pd.api.types.is_integer_dtype(s):
                t = "categorielle"
            else:
                t = "numerique"
        else:
            if k == s.notna().sum() and k > max_modalites:
                t = "identifiant"
            elif k <= max_modalites or isinstance(s.dtype, pd.CategoricalDtype):
                t = "categorielle"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    essai = pd.to_datetime(s.dropna().head(200), errors="coerce", dayfirst=True)
                t = "date" if essai.notna().mean() > 0.9 else "texte"
        lignes.append({"variable": c, "type_detecte": t, "n_modalites": k, "dtype": str(s.dtype)})
    return pd.DataFrame(lignes)


def convertir_dates(df: pd.DataFrame, colonnes: Sequence[str], format: str | None = None) -> pd.DataFrame:
    """Convertit des colonnes texte en dates (format français jour/mois/année par défaut)."""
    colonnes = en_liste(colonnes)
    verifier_colonnes(df, colonnes)
    out = df.copy()
    for c in colonnes:
        out[c] = pd.to_datetime(out[c], format=format, dayfirst=format is None, errors="coerce")
    return out


def calculer_age(date_naissance, date_reference) -> pd.Series:
    """Âge révolu (en années entières) à une date de référence."""
    dn = pd.to_datetime(pd.Series(date_naissance), errors="coerce", dayfirst=True)
    if isinstance(date_reference, (str, pd.Timestamp)) or np.isscalar(date_reference):
        dr = pd.Series(pd.Timestamp(date_reference), index=dn.index)
    else:
        dr = pd.to_datetime(pd.Series(date_reference, index=dn.index), errors="coerce", dayfirst=True)
    age = dr.dt.year - dn.dt.year
    pas_encore = (dr.dt.month < dn.dt.month) | ((dr.dt.month == dn.dt.month) & (dr.dt.day < dn.dt.day))
    return (age - pas_encore.astype(int)).astype("Int64")


def tranches_age(age, bornes: Sequence[int] = TRANCHES_AGE_ML, libelles: Sequence[str] | None = None) -> pd.Series:
    """Découpe un âge en tranches (catégorie ordonnée).

    Parameters
    ----------
    bornes : sequence of int
        Bornes inférieures incluses ; la dernière ouvre la tranche « et + ».
        Défaut : 16-17, 18-21, 22-25, 26 et +. Les âges sous la première borne
        forment une tranche « moins de … ».

    Examples
    --------
    >>> tranches_age(pd.Series([16, 19, 25, 27])).tolist()
    ['16-17 ans', '18-21 ans', '22-25 ans', '26 ans et +']
    """
    bornes = list(bornes)
    a = pd.to_numeric(pd.Series(age), errors="coerce")
    coupures = [-np.inf, *bornes, np.inf]
    if libelles is None:
        libelles = [f"moins de {bornes[0]} ans"]
        libelles += [f"{bornes[i]}-{bornes[i + 1] - 1} ans" for i in range(len(bornes) - 1)]
        libelles += [f"{bornes[-1]} ans et +"]
    elif len(libelles) == len(bornes):
        libelles = [f"moins de {bornes[0]} ans", *libelles]
    out = pd.cut(a, coupures, right=False, labels=libelles, ordered=True)
    if not (a < bornes[0]).any():
        out = out.cat.remove_categories([libelles[0]])
    return out


def discretiser(x, n_classes: int = 4, methode: str = "quantiles", libelles: Sequence[str] | None = None,
                decimales: int = 1) -> pd.Series:
    """Découpe une variable numérique en classes.

    Parameters
    ----------
    methode : {"quantiles", "egales"}
        Classes d'effectifs égaux (quartiles, quintiles…) ou d'amplitudes égales.
    """
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if methode == "quantiles":
        return pd.qcut(s, n_classes, labels=libelles, duplicates="drop", precision=decimales)
    if methode == "egales":
        return pd.cut(s, n_classes, labels=libelles, precision=decimales)
    raise ValueError("methode : 'quantiles' ou 'egales'.")


def duree_entre(debut, fin, unite: str = "jours") -> pd.Series:
    """Durée entre deux dates en ``jours``, ``semaines``, ``mois`` ou ``annees``."""
    facteurs = {"jours": 1.0, "semaines": 7.0, "mois": 365.25 / 12, "annees": 365.25}
    if unite not in facteurs:
        raise ValueError(f"unite parmi {list(facteurs)}")
    d1 = pd.to_datetime(pd.Series(debut), errors="coerce", dayfirst=True)
    d2 = pd.to_datetime(pd.Series(fin, index=d1.index) if not np.isscalar(fin) else
                        pd.Series(pd.Timestamp(fin), index=d1.index), errors="coerce", dayfirst=True)
    return (d2 - d1).dt.days / facteurs[unite]


def recoder(serie: pd.Series, correspondance: Mapping, autre: object = None,
            ordre: Sequence | None = None) -> pd.Series:
    """Recode les modalités d'une variable.

    Parameters
    ----------
    correspondance : dict
        ``{ancienne: nouvelle}``. Une clé peut être un tuple/liste pour regrouper
        plusieurs modalités : ``{("CDI", "CDD > 6 mois"): "Emploi durable"}``.
    autre : optional
        Valeur des modalités non listées (défaut : conservées telles quelles).
    ordre : list, optional
        Ordre des modalités : renvoie une catégorie ordonnée.
    """
    plat = {}
    for k, v in correspondance.items():
        if isinstance(k, (tuple, list, set, frozenset)):
            for kk in k:
                plat[kk] = v
        else:
            plat[k] = v
    s = pd.Series(serie)
    out = s.map(plat)
    reste = out.isna() & s.notna()
    out = out.astype(object)
    out[reste] = autre if autre is not None else s[reste]
    if ordre is not None:
        out = pd.Categorical(out, categories=list(ordre), ordered=True)
        return pd.Series(out, index=s.index, name=s.name)
    return out.rename(s.name)


def regrouper_modalites_rares(serie: pd.Series, seuil: float = 0.05, libelle: str = "Autre",
                              min_effectif: int | None = None) -> pd.Series:
    """Regroupe les modalités peu fréquentes (part < ``seuil`` ou effectif < ``min_effectif``).

    Évite les modalités à faible effectif qui rendent les modèles instables et
    les tests du χ² non valides.
    """
    s = pd.Series(serie)
    freq = s.value_counts(normalize=True)
    eff = s.value_counts()
    rares = set(freq[freq < seuil].index)
    if min_effectif is not None:
        rares |= set(eff[eff < min_effectif].index)
    out = s.astype(object).where(~s.isin(rares), libelle)
    return out


def convertir_binaire(serie: pd.Series, positives: Sequence | object = None) -> pd.Series:
    """Convertit en 0/1 (``Int64``, manquants conservés).

    Sans ``positives``, reconnaît les réponses usuelles : oui/non, o/n, vrai/faux,
    true/false, 1/0, x/vide.
    """
    s = pd.Series(serie)
    if positives is not None:
        pos = set(positives) if isinstance(positives, (list, tuple, set)) else {positives}
        out = s.isin(pos).astype("Int64")
        out[s.isna()] = pd.NA
        return out
    vrais = {"oui", "o", "vrai", "true", "1", "x", "yes", "y"}
    faux = {"non", "n", "faux", "false", "0", "no", ""}
    norm = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out[norm.isin(vrais)] = 1
    out[norm.isin(faux)] = 0
    out[s.isna()] = pd.NA
    inconnus = norm[out.isna() & s.notna()].unique()
    if len(inconnus):
        raise ValueError(f"Valeurs non reconnues : {list(inconnus)[:10]} ; précisez `positives=`.")
    return out


def imputer(df: pd.DataFrame, strategie: Mapping[str, object] | str = "auto") -> pd.DataFrame:
    """Imputation simple des valeurs manquantes.

    Parameters
    ----------
    strategie : "auto" ou dict
        ``"auto"`` : médiane pour les numériques, modalité « Non renseigné »
        pour les catégorielles (souvent préférable à la modalité majoritaire en
        sociologie : le non-renseignement est une information). Un dict permet
        de préciser par colonne : ``"mediane"``, ``"moyenne"``, ``"mode"``,
        ``"non_renseigne"`` ou une valeur fixe.

    .. warning::
       L'imputation simple sous-estime l'incertitude ; pour une analyse
       inférentielle exigeante, préférer l'imputation multiple.
    """
    out = df.copy()
    if isinstance(strategie, str):
        if strategie != "auto":
            raise ValueError("strategie : 'auto' ou dict.")
        strategie = {c: ("mediane" if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])
                         else "non_renseigne") for c in out.columns if out[c].isna().any()}
    for c, st in strategie.items():
        s = out[c]
        if st == "mediane":
            val = s.median()
        elif st == "moyenne":
            val = s.mean()
        elif st == "mode":
            val = s.mode().iloc[0]
        elif st == "non_renseigne":
            val = "Non renseigné"
            if isinstance(s.dtype, pd.CategoricalDtype):
                if val not in s.cat.categories:
                    out[c] = s.cat.add_categories([val])
                out[c] = out[c].fillna(val)
                continue
            out[c] = s.astype(object)
        else:
            val = st
        out[c] = out[c].fillna(val)
    return out


def ordonner_modalites(serie: pd.Series, ordre: Sequence) -> pd.Series:
    """Transforme une variable en catégorie ordonnée (contrôle l'ordre des tableaux et graphiques)."""
    s = pd.Series(serie)
    inconnues = set(s.dropna().unique()) - set(ordre)
    if inconnues:
        raise ValueError(f"Modalités absentes de `ordre` : {sorted(map(str, inconnues))}")
    return pd.Series(pd.Categorical(s, categories=list(ordre), ordered=True), index=s.index, name=s.name)
