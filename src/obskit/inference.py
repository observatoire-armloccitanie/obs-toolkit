"""Tests statistiques et intervalles de confiance.

Fonctions « clés en main » qui choisissent un test adapté, renvoient la
statistique, la p-valeur **et une taille d'effet** (indispensable avec de
gros effectifs, où tout devient significatif).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import verifier_colonnes

__all__ = [
    "intervalle_proportion",
    "intervalle_moyenne",
    "test_independance",
    "comparer_groupes",
    "comparer_proportions",
    "test_normalite",
    "corriger_pvaleurs",
    "taille_echantillon_proportion",
    "v_cramer",
]


def intervalle_proportion(k: float, n: float, alpha: float = 0.05, methode: str = "wilson") -> tuple[float, float]:
    """Intervalle de confiance d'une proportion ``k / n``.

    Parameters
    ----------
    methode : {"wilson", "clopper-pearson", "normale"}
        Wilson (défaut) est fiable même pour de petits effectifs ou des
        proportions proches de 0 ou 1 ; Clopper-Pearson est exact mais conservateur.

    Examples
    --------
    >>> lo, hi = intervalle_proportion(45, 120)
    >>> print(round(lo, 3), round(hi, 3))
    0.294 0.464
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    if methode == "wilson":
        centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        demi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
        return (max(0.0, centre - demi), min(1.0, centre + demi))
    if methode == "clopper-pearson":
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return (float(lo), float(hi))
    if methode == "normale":
        demi = z * np.sqrt(p * (1 - p) / n)
        return (max(0.0, p - demi), min(1.0, p + demi))
    raise ValueError("methode : 'wilson', 'clopper-pearson' ou 'normale'.")


def intervalle_moyenne(x, alpha: float = 0.05, poids=None) -> tuple[float, float, float]:
    """Moyenne et intervalle de confiance (loi de Student ; approximation de Kish si pondéré).

    Returns
    -------
    (moyenne, borne_basse, borne_haute)
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if poids is None:
        n = len(x)
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(n)
        q = stats.t.ppf(1 - alpha / 2, n - 1)
        return (float(m), float(m - q * se), float(m + q * se))
    w = np.asarray(poids, dtype=float)
    m = np.average(x, weights=w)
    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    var = np.average((x - m) ** 2, weights=w) * n_eff / (n_eff - 1)
    se = np.sqrt(var / n_eff)
    q = stats.t.ppf(1 - alpha / 2, n_eff - 1)
    return (float(m), float(m - q * se), float(m + q * se))


def v_cramer(x, y, correction: bool = True) -> float:
    """V de Cramér entre deux variables catégorielles (0 = indépendance, 1 = liaison parfaite).

    ``correction=True`` applique la correction de biais de Bergsma (2013).
    """
    tab = pd.crosstab(pd.Series(x), pd.Series(y))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = chi2 / n
    if correction:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def test_independance(
    df: pd.DataFrame, x: str, y: str, poids: str | None = None, exact: bool | None = None,
) -> dict:
    """Test d'indépendance entre deux variables catégorielles (χ² ou Fisher exact).

    Parameters
    ----------
    x, y : str
        Colonnes catégorielles.
    poids : str, optional
        Pondération : le tableau est pondéré puis ramené à l'effectif réel
        (approximation ; pour un plan de sondage complexe, la p-valeur reste indicative).
    exact : bool, optional
        Force (``True``) ou interdit (``False``) le test exact de Fisher. Par
        défaut, Fisher est utilisé pour un tableau 2x2 dont un effectif
        théorique est < 5.

    Returns
    -------
    dict
        ``test``, ``statistique``, ``ddl``, ``p_valeur``, ``v_cramer``,
        ``part_effectifs_theoriques_inf_5``, ``tableau``, ``residus_standardises``.
    """
    verifier_colonnes(df, [x, y, poids])
    d = df[[c for c in (x, y, poids) if c]].dropna()
    if poids:
        tab = d.pivot_table(index=x, columns=y, values=poids, aggfunc="sum", fill_value=0, observed=True)
        tab = tab * len(d) / tab.to_numpy().sum()
    else:
        tab = pd.crosstab(d[x], d[y])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError("Chaque variable doit avoir au moins deux modalités observées.")
    chi2, p, ddl, att = stats.chi2_contingency(tab, correction=False)
    part_inf5 = float(np.mean(att < 5))
    resultat = {"test": "chi2", "statistique": float(chi2), "ddl": int(ddl), "p_valeur": float(p)}
    if exact is None:
        exact = tab.shape == (2, 2) and att.min() < 5
    if exact:
        if tab.shape != (2, 2):
            warnings.warn("Test exact limité aux tableaux 2x2 : p-valeur du χ² conservée.", stacklevel=2)
        else:
            oddsr, pf = stats.fisher_exact(np.round(tab.to_numpy()).astype(int))
            resultat.update({"test": "fisher_exact", "statistique": float(oddsr), "ddl": None, "p_valeur": float(pf)})
    elif part_inf5 > 0.2:
        warnings.warn(f"{part_inf5:.0%} des effectifs théoriques < 5 : χ² peu fiable "
                      "(regrouper des modalités).", stacklevel=2)
    residus = (tab - att) / np.sqrt(att)
    if poids:
        v = float(np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1))))
    else:
        v = v_cramer(d[x], d[y])
    resultat.update({
        "v_cramer": v,
        "part_effectifs_theoriques_inf_5": part_inf5,
        "tableau": tab,
        "residus_standardises": residus,
    })
    return resultat


def _cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else np.nan


def comparer_groupes(
    df: pd.DataFrame, variable: str, groupe: str, parametrique: bool | None = None, alpha_normalite: float = 0.05,
) -> dict:
    """Compare une variable quantitative entre groupes, avec choix automatique du test.

    * 2 groupes : t de Welch (paramétrique) ou Mann-Whitney ;
    * ≥ 3 groupes : ANOVA de Welch (paramétrique) ou Kruskal-Wallis.

    Parameters
    ----------
    parametrique : bool, optional
        ``None`` (défaut) : test paramétrique si chaque groupe compte au moins
        30 individus ou si la normalité n'est pas rejetée (Shapiro-Wilk) ;
        sinon test non paramétrique.

    Returns
    -------
    dict
        ``test``, ``statistique``, ``p_valeur``, ``taille_effet`` (d de Cohen,
        r rang-bisérial, η² ou ε²), ``nom_taille_effet`` et ``resume_groupes``.
    """
    verifier_colonnes(df, [variable, groupe])
    d = df[[variable, groupe]].dropna()
    groupes = {g: s.to_numpy(dtype=float) for g, s in d.groupby(groupe, observed=True)[variable]}
    groupes = {g: v for g, v in groupes.items() if len(v) > 0}
    if len(groupes) < 2:
        raise ValueError("Il faut au moins deux groupes non vides.")
    vals = list(groupes.values())
    if parametrique is None:
        normal = all(len(v) >= 30 or (len(v) >= 3 and stats.shapiro(v).pvalue > alpha_normalite) for v in vals)
        parametrique = normal
    resume = d.groupby(groupe, observed=True)[variable].agg(["count", "mean", "std", "median"]).rename(
        columns={"count": "n", "mean": "moyenne", "std": "ecart_type", "median": "mediane"})
    if len(vals) == 2:
        a, b = vals
        if parametrique:
            r = stats.ttest_ind(a, b, equal_var=False)
            out = {"test": "t de Welch", "statistique": float(r.statistic), "p_valeur": float(r.pvalue),
                   "taille_effet": _cohen_d(a, b), "nom_taille_effet": "d de Cohen"}
        else:
            r = stats.mannwhitneyu(a, b, alternative="two-sided")
            rbis = 2 * r.statistic / (len(a) * len(b)) - 1
            out = {"test": "Mann-Whitney", "statistique": float(r.statistic), "p_valeur": float(r.pvalue),
                   "taille_effet": float(rbis), "nom_taille_effet": "r rang-bisérial"}
    else:
        n = sum(len(v) for v in vals)
        k = len(vals)
        if parametrique:
            # ANOVA de Welch (variances inégales)
            m = np.array([v.mean() for v in vals])
            s2 = np.array([v.var(ddof=1) for v in vals])
            nn = np.array([len(v) for v in vals])
            w = nn / s2
            mw = np.sum(w * m) / w.sum()
            A = np.sum(w * (m - mw) ** 2) / (k - 1)
            tmp = np.sum((1 - w / w.sum()) ** 2 / (nn - 1))
            B = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
            F = A / B
            ddl2 = (k ** 2 - 1) / (3 * tmp)
            p = float(stats.f.sf(F, k - 1, ddl2))
            gm = np.concatenate(vals).mean()
            ss_inter = sum(len(v) * (v.mean() - gm) ** 2 for v in vals)
            ss_tot = sum(((v - gm) ** 2).sum() for v in vals)
            out = {"test": "ANOVA de Welch", "statistique": float(F), "p_valeur": p,
                   "taille_effet": float(ss_inter / ss_tot), "nom_taille_effet": "eta²"}
        else:
            r = stats.kruskal(*vals)
            eps2 = (r.statistic - k + 1) / (n - k)
            out = {"test": "Kruskal-Wallis", "statistique": float(r.statistic), "p_valeur": float(r.pvalue),
                   "taille_effet": float(max(eps2, 0)), "nom_taille_effet": "epsilon²"}
    out["resume_groupes"] = resume
    return out


def comparer_proportions(k1: float, n1: float, k2: float, n2: float, alpha: float = 0.05) -> dict:
    """Test z de comparaison de deux proportions (ex. taux d'accès à l'emploi QPV vs hors QPV).

    Returns
    -------
    dict
        ``p1``, ``p2``, ``difference``, IC de la différence, ``z``, ``p_valeur``,
        ``rapport`` (p1/p2) et ``h_cohen`` (taille d'effet).
    """
    p1, p2 = k1 / n1, k2 / n2
    pc = (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pc * (1 - pc) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 > 0 else np.nan
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    q = stats.norm.ppf(1 - alpha / 2)
    return {"p1": p1, "p2": p2, "difference": p1 - p2, "ic_bas": p1 - p2 - q * se, "ic_haut": p1 - p2 + q * se,
            "z": float(z), "p_valeur": float(2 * stats.norm.sf(abs(z))),
            "rapport": p1 / p2 if p2 > 0 else np.nan,
            "h_cohen": float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))}


def test_normalite(x) -> dict:
    """Tests de normalité : Shapiro-Wilk (n ≤ 5000) et D'Agostino-Pearson, avec asymétrie et aplatissement."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    out = {"n": len(x), "asymetrie": float(stats.skew(x)), "aplatissement": float(stats.kurtosis(x))}
    if 3 <= len(x) <= 5000:
        out["shapiro_p"] = float(stats.shapiro(x).pvalue)
    if len(x) >= 20:
        out["dagostino_p"] = float(stats.normaltest(x).pvalue)
    return out


def corriger_pvaleurs(p: Sequence[float], methode: str = "holm") -> np.ndarray:
    """Correction pour tests multiples.

    Parameters
    ----------
    methode : {"bonferroni", "holm", "fdr_bh"}
        ``holm`` contrôle le risque familial (plus puissant que Bonferroni) ;
        ``fdr_bh`` (Benjamini-Hochberg) contrôle le taux de fausses découvertes.
    """
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    if methode == "bonferroni":
        adj = np.minimum(pv * m, 1)
    elif methode == "holm":
        o = np.argsort(pv)
        adj_o = np.maximum.accumulate((m - np.arange(m)) * pv[o])
        adj = np.empty(m)
        adj[o] = np.minimum(adj_o, 1)
    elif methode == "fdr_bh":
        o = np.argsort(pv)[::-1]
        rangs = m - np.arange(m)
        adj_o = np.minimum.accumulate(pv[o] * m / rangs)
        adj = np.empty(m)
        adj[o] = np.minimum(adj_o, 1)
    else:
        raise ValueError("methode : 'bonferroni', 'holm' ou 'fdr_bh'.")
    out[ok] = adj
    return out


def taille_echantillon_proportion(marge: float, p: float = 0.5, alpha: float = 0.05,
                                  population: int | None = None) -> int:
    """Taille d'échantillon pour estimer une proportion avec une marge d'erreur donnée.

    Parameters
    ----------
    marge : float
        Demi-largeur souhaitée de l'IC (ex. 0.03 pour ± 3 points).
    p : float
        Proportion attendue (0,5 = cas le plus défavorable).
    population : int, optional
        Taille de la population (correction pour population finie), ex. nombre
        de jeunes d'une cohorte pour une enquête.

    Examples
    --------
    >>> taille_echantillon_proportion(0.05)
    385
    """
    z = stats.norm.ppf(1 - alpha / 2)
    n0 = z ** 2 * p * (1 - p) / marge ** 2
    if population:
        n0 = n0 / (1 + (n0 - 1) / population)
    return int(np.ceil(n0))
