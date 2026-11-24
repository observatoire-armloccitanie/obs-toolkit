"""Corrélations et mesures d'association entre variables de tous types.

* numérique x numérique : Pearson, Spearman, Kendall ;
* catégorielle x catégorielle : V de Cramér ;
* numérique x catégorielle : rapport de corrélation η (eta).

:func:`associations_avec_cible` classe toutes les variables selon la force de
leur lien avec une variable cible : un bon point de départ avant de modéliser.

.. note::
   Une corrélation ne prouve pas une relation causale : voir le module
   :mod:`obskit.causalite` pour les méthodes d'estimation d'effets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import inference
from ._utils import en_liste, est_categorielle, explicatives_par_defaut, verifier_colonnes

__all__ = [
    "matrice_correlation",
    "correlation_partielle",
    "rapport_correlation",
    "matrice_association",
    "associations_avec_cible",
]


def matrice_correlation(
    df: pd.DataFrame,
    variables: Sequence[str] | None = None,
    methode: str = "pearson",
    correction: str | None = "holm",
    format_long: bool = False,
) -> dict | pd.DataFrame:
    """Matrice de corrélation avec p-valeurs et effectifs par paire.

    Parameters
    ----------
    variables : list of str, optional
        Variables numériques (défaut : toutes).
    methode : {"pearson", "spearman", "kendall"}
        Spearman est robuste aux valeurs extrêmes et aux relations monotones non linéaires.
    correction : {"holm", "bonferroni", "fdr_bh", None}
        Correction des p-valeurs pour tests multiples.
    format_long : bool
        Si ``True``, renvoie un DataFrame long (une ligne par paire) trié par
        valeur absolue de r décroissante ; sinon un dict ``{"r", "p_valeur", "n"}`` de matrices.
    """
    if variables is None:
        variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_bool_dtype(df[c])]
    variables = en_liste(variables)
    verifier_colonnes(df, variables)
    k = len(variables)
    R = np.eye(k)
    P = np.zeros((k, k))
    N = np.zeros((k, k), dtype=int)
    fonctions = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
    if methode not in fonctions:
        raise ValueError("methode : 'pearson', 'spearman' ou 'kendall'.")
    paires = []
    for i in range(k):
        N[i, i] = int(df[variables[i]].notna().sum())
        for j in range(i + 1, k):
            d = df[[variables[i], variables[j]]].dropna().astype(float)
            N[i, j] = N[j, i] = len(d)
            if len(d) < 3 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                res = fonctions[methode](d.iloc[:, 0], d.iloc[:, 1])
                r, p = float(res[0]), float(res[1])
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
            paires.append((i, j))
    if correction and paires:
        padj = inference.corriger_pvaleurs([P[i, j] for i, j in paires], correction)
        for (i, j), pa in zip(paires, padj):
            P[i, j] = P[j, i] = pa
    Rdf = pd.DataFrame(R, index=variables, columns=variables)
    Pdf = pd.DataFrame(P, index=variables, columns=variables)
    Ndf = pd.DataFrame(N, index=variables, columns=variables)
    if format_long:
        out = pd.DataFrame([{"variable_1": variables[i], "variable_2": variables[j], "r": R[i, j],
                             "p_valeur": P[i, j], "n": N[i, j]} for i, j in paires])
        if len(out):
            out = out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return out
    return {"r": Rdf, "p_valeur": Pdf, "n": Ndf}


def correlation_partielle(
    df: pd.DataFrame, x: str, y: str, controles: Sequence[str], methode: str = "pearson",
) -> dict:
    """Corrélation partielle entre ``x`` et ``y`` en neutralisant des variables de contrôle.

    Calculée sur les résidus des régressions linéaires de ``x`` et ``y`` sur les
    contrôles (sur les rangs si ``methode="spearman"``). Les contrôles
    catégoriels sont codés en indicatrices.
    """
    from ._encodage import Encodeur

    controles = en_liste(controles)
    verifier_colonnes(df, [x, y, *controles])
    d = df[[x, y, *controles]].dropna()
    xs, ys = d[x].astype(float), d[y].astype(float)
    if methode == "spearman":
        xs, ys = xs.rank(), ys.rank()
    Z = Encodeur(controles).fit_transform(d).to_numpy(dtype=float)
    rx = xs.to_numpy() - Z @ np.linalg.lstsq(Z, xs.to_numpy(), rcond=None)[0]
    ry = ys.to_numpy() - Z @ np.linalg.lstsq(Z, ys.to_numpy(), rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, q = len(d), Z.shape[1] - 1
    ddl = n - 2 - q
    t = r * np.sqrt(ddl / (1 - r ** 2))
    return {"r_partiel": r, "n": n, "ddl": ddl, "p_valeur": float(2 * stats.t.sf(abs(t), ddl))}


def rapport_correlation(categorie, valeur) -> float:
    """Rapport de corrélation η entre une variable catégorielle et une variable numérique.

    η² est la part de variance de ``valeur`` expliquée par ``categorie`` (0 à 1).
    """
    d = pd.DataFrame({"c": pd.Series(categorie).to_numpy(), "v": pd.Series(valeur).to_numpy()}).dropna()
    v = d["v"].astype(float)
    if len(d) == 0 or v.var() == 0:
        return np.nan
    moy = v.mean()
    groupes = d.groupby("c", observed=True)["v"]
    ss_inter = float((groupes.count() * (groupes.mean() - moy) ** 2).sum())
    ss_tot = float(((v - moy) ** 2).sum())
    return float(np.sqrt(ss_inter / ss_tot))


def _association(s1: pd.Series, s2: pd.Series, methode_num: str) -> tuple[float, str]:
    c1, c2 = est_categorielle(s1), est_categorielle(s2)
    d = pd.DataFrame({"a": s1.to_numpy(), "b": s2.to_numpy()}).dropna()
    if len(d) < 3:
        return np.nan, ""
    if c1 and c2:
        return inference.v_cramer(d["a"], d["b"]), "V de Cramér"
    if c1 and not c2:
        return rapport_correlation(d["a"], d["b"]), "eta"
    if c2 and not c1:
        return rapport_correlation(d["b"], d["a"]), "eta"
    f = stats.spearmanr if methode_num == "spearman" else stats.pearsonr
    return abs(float(f(d["a"].astype(float), d["b"].astype(float))[0])), f"|{methode_num}|"


def matrice_association(
    df: pd.DataFrame, variables: Sequence[str] | None = None, methode_num: str = "spearman",
) -> pd.DataFrame:
    """Matrice d'association (0 à 1) entre variables de types mélangés.

    Pratique pour repérer d'un coup d'œil les variables redondantes avant une
    régression. Les valeurs ne sont comparables qu'approximativement entre types de paires.
    """
    variables = en_liste(variables) or list(df.columns)
    verifier_colonnes(df, variables)
    k = len(variables)
    M = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            M[i, j] = M[j, i] = _association(df[variables[i]], df[variables[j]], methode_num)[0]
    return pd.DataFrame(M, index=variables, columns=variables)


def associations_avec_cible(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    methode_num: str = "spearman",
    correction: str | None = "holm",
) -> pd.DataFrame:
    """Force et significativité du lien entre chaque variable et une cible.

    Le test est choisi selon les types : χ² (cat x cat), Kruskal-Wallis ou
    Mann-Whitney (cat x num), corrélation de Spearman/Pearson (num x num).

    Returns
    -------
    DataFrame trié par force d'association décroissante, colonnes
    ``variable``, ``mesure``, ``association``, ``test``, ``p_valeur``, ``n``.
    """
    explicatives = explicatives_par_defaut(df, cible, explicatives)
    verifier_colonnes(df, [cible, *explicatives])
    lignes = []
    cible_cat = est_categorielle(df[cible])
    for v in explicatives:
        d = df[[cible, v]].dropna()
        if len(d) < 3 or d[v].nunique() < 2:
            continue
        assoc, mesure = _association(d[v], d[cible], methode_num)
        v_cat = est_categorielle(d[v])
        try:
            if cible_cat and v_cat:
                t = inference.test_independance(d, v, cible)
                test, p = t["test"], t["p_valeur"]
            elif cible_cat or v_cat:
                cat, num = (cible, v) if cible_cat else (v, cible)
                t = inference.comparer_groupes(d, num, cat, parametrique=False)
                test, p = t["test"], t["p_valeur"]
            else:
                f = stats.spearmanr if methode_num == "spearman" else stats.pearsonr
                test, p = methode_num, float(f(d[v].astype(float), d[cible].astype(float))[1])
        except ValueError:
            test, p = "", np.nan
        lignes.append({"variable": v, "mesure": mesure, "association": assoc, "test": test,
                       "p_valeur": p, "n": len(d)})
    out = pd.DataFrame(lignes)
    if len(out) and correction:
        out["p_valeur_corrigee"] = inference.corriger_pvaleurs(out["p_valeur"], correction)
    return out.sort_values("association", ascending=False).reset_index(drop=True) if len(out) else out
