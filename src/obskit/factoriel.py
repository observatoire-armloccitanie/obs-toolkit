"""Analyses factorielles et classification : ACP, AFC, ACM, CAH.

La démarche « ACM + CAH » est classique en sociologie quantitative pour
construire des typologies (profils de jeunes, de parcours…) :

1. :func:`acm` résume un ensemble de variables qualitatives en quelques axes ;
2. :func:`classification` (CAH de Ward) regroupe les individus à partir de
   leurs coordonnées sur les premiers axes ;
3. :func:`caracteriser_classes` décrit chaque classe par ses modalités sur-représentées.

Les calculs reposent uniquement sur numpy/scipy et suivent les conventions de
FactoMineR (coordonnées, contributions en %, cos²).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy

from ._utils import en_liste, verifier_colonnes

__all__ = ["acp", "afc", "acm", "ResultatFactoriel", "classification", "caracteriser_classes"]


@dataclass
class ResultatFactoriel:
    """Résultat d'une analyse factorielle.

    Attributes
    ----------
    valeurs_propres : DataFrame
        ``valeur_propre``, ``pct_inertie``, ``pct_cumule`` par axe.
    individus : dict of DataFrame
        ``coord``, ``contrib`` (%), ``cos2`` des individus (ou lignes pour l'AFC).
    variables : dict of DataFrame
        Idem pour les variables (ACP), modalités (ACM) ou colonnes (AFC).
    supplementaires : dict
        Coordonnées des éléments supplémentaires éventuels.
    """

    methode: str
    valeurs_propres: pd.DataFrame
    individus: dict
    variables: dict
    supplementaires: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict, repr=False)

    def tracer(self, axes=(1, 2), quoi: str = "variables", ax=None, **kwargs):
        """Plan factoriel (voir :func:`obskit.visualisation.plan_factoriel`)."""
        from .visualisation import plan_factoriel

        return plan_factoriel(self, axes=axes, quoi=quoi, ax=ax, **kwargs)


def _tableau_vp(vp: np.ndarray) -> pd.DataFrame:
    pct = vp / vp.sum() * 100
    return pd.DataFrame({"valeur_propre": vp, "pct_inertie": pct, "pct_cumule": np.cumsum(pct)},
                        index=pd.Index([f"Dim.{i + 1}" for i in range(len(vp))], name="axe"))


def _noms_axes(k):
    return [f"Dim.{i + 1}" for i in range(k)]


def acp(
    df: pd.DataFrame, variables: Sequence[str] | None = None, n_axes: int = 5, reduire: bool = True,
    poids: str | None = None, variables_sup: Sequence[str] | None = None,
) -> ResultatFactoriel:
    """Analyse en composantes principales (normée par défaut).

    Parameters
    ----------
    variables : list of str, optional
        Variables numériques actives (défaut : toutes les numériques).
    n_axes : int
        Nombre d'axes conservés dans les sorties.
    reduire : bool
        ACP normée (variables centrées-réduites) : recommandé si les unités diffèrent.
    poids : str, optional
        Poids des individus.
    variables_sup : list of str, optional
        Variables numériques supplémentaires (corrélations avec les axes).
    """
    if variables is None:
        variables = [c for c in df.select_dtypes("number").columns if c not in en_liste(variables_sup) and c != poids]
    variables = en_liste(variables)
    verifier_colonnes(df, [*variables, *en_liste(variables_sup), poids])
    d = df[variables + en_liste(variables_sup) + ([poids] if poids else [])].dropna()
    X = d[variables].to_numpy(dtype=float)
    w = d[poids].to_numpy(dtype=float) if poids else np.ones(len(d))
    w = w / w.sum()
    moy = w @ X
    Xc = X - moy
    et = np.sqrt(w @ Xc ** 2)
    Z = Xc / et if reduire else Xc
    U, s, Vt = np.linalg.svd(Z * np.sqrt(w)[:, None], full_matrices=False)
    vp = s ** 2
    k = min(n_axes, len(vp))
    F = (U[:, :k] * s[:k]) / np.sqrt(w)[:, None]  # coordonnées des individus
    G = Vt[:k].T * s[:k]  # coordonnées des variables (corrélations si réduites)
    axes = _noms_axes(k)
    d2_ind = np.sum(Z ** 2, axis=1)
    ind = {"coord": pd.DataFrame(F, index=d.index, columns=axes),
           "contrib": pd.DataFrame(w[:, None] * F ** 2 / vp[:k] * 100, index=d.index, columns=axes),
           "cos2": pd.DataFrame(F ** 2 / d2_ind[:, None], index=d.index, columns=axes)}
    d2_var = np.ones(len(variables)) if reduire else w @ Z ** 2
    var = {"coord": pd.DataFrame(G, index=variables, columns=axes),
           "contrib": pd.DataFrame(G ** 2 / vp[:k] * 100, index=variables, columns=axes),
           "cos2": pd.DataFrame(G ** 2 / d2_var[:, None], index=variables, columns=axes)}
    sup = {}
    if variables_sup:
        cor = {}
        for v in en_liste(variables_sup):
            x = d[v].to_numpy(dtype=float)
            xc = (x - w @ x) / np.sqrt(w @ (x - w @ x) ** 2)
            cor[v] = [(w * xc) @ (F[:, j] / np.sqrt(vp[j])) for j in range(k)]
        sup["variables"] = pd.DataFrame(cor, index=axes).T
    return ResultatFactoriel("ACP", _tableau_vp(vp), ind, var, sup,
                             {"moyennes": moy, "ecarts_types": et, "reduire": reduire})


def afc(tableau: pd.DataFrame, n_axes: int = 5) -> ResultatFactoriel:
    """Analyse factorielle des correspondances d'un tableau de contingence.

    Parameters
    ----------
    tableau : DataFrame
        Tableau de contingence (effectifs), par ex. ``pd.crosstab(df.departement, df.situation_sortie)``.

    Notes
    -----
    L'inertie totale vaut χ²/n. Les lignes sont stockées dans ``individus`` et
    les colonnes dans ``variables``.
    """
    N = tableau.to_numpy(dtype=float)
    n = N.sum()
    P = N / n
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    vp = s ** 2
    nz = vp > 1e-12
    vp, U, s, Vt = vp[nz], U[:, nz], s[nz], Vt[nz]
    k = min(n_axes, len(vp))
    F = U[:, :k] * s[:k] / np.sqrt(r)[:, None]
    G = Vt[:k].T * s[:k] / np.sqrt(c)[:, None]
    axes = _noms_axes(k)
    d2r = np.sum(((P / r[:, None]) - c) ** 2 / c, axis=1)
    d2c = np.sum(((P / c[None, :]) - r[:, None]) ** 2 / r[:, None], axis=0)
    lignes = {"coord": pd.DataFrame(F, index=tableau.index, columns=axes),
              "contrib": pd.DataFrame(r[:, None] * F ** 2 / vp[:k] * 100, index=tableau.index, columns=axes),
              "cos2": pd.DataFrame(F ** 2 / d2r[:, None], index=tableau.index, columns=axes)}
    cols = {"coord": pd.DataFrame(G, index=tableau.columns, columns=axes),
            "contrib": pd.DataFrame(c[:, None] * G ** 2 / vp[:k] * 100, index=tableau.columns, columns=axes),
            "cos2": pd.DataFrame(G ** 2 / d2c[:, None], index=tableau.columns, columns=axes)}
    chi2 = stats.chi2_contingency(N, correction=False)[0]
    return ResultatFactoriel("AFC", _tableau_vp(vp), lignes, cols,
                             details={"inertie_totale": vp.sum(), "chi2": chi2, "n": n})


def acm(
    df: pd.DataFrame, variables: Sequence[str], n_axes: int = 5,
    variables_sup: Sequence[str] | None = None, poids: str | None = None,
    correction_benzecri: bool = False,
) -> ResultatFactoriel:
    """Analyse des correspondances multiples (variables qualitatives).

    Parameters
    ----------
    variables : list of str
        Variables qualitatives actives (les manquants deviennent une modalité « NA »).
    variables_sup : list of str, optional
        Variables qualitatives supplémentaires : leurs modalités sont projetées
        (barycentres) sans participer à la construction des axes - par ex.
        projeter ``situation_sortie`` sur un espace des profils sociaux.
    poids : str, optional
        Poids des individus.
    correction_benzecri : bool
        Ajoute les taux d'inertie corrigés de Benzécri (plus réalistes que les
        taux bruts, toujours faibles en ACM).

    Returns
    -------
    ResultatFactoriel
        ``variables`` décrit les modalités (index ``variable=modalite``) avec
        en plus ``v_test`` (significativité de la position de la modalité).
    """
    variables = en_liste(variables)
    sup = en_liste(variables_sup)
    verifier_colonnes(df, [*variables, *sup, poids])
    d = df[variables + sup].astype(object).where(df[variables + sup].notna(), "NA")
    w = df[poids].to_numpy(dtype=float) if poids else np.ones(len(df))
    w = w / w.sum()
    Q = len(variables)
    Z = pd.get_dummies(d[variables], prefix_sep="=").astype(float)
    Zv = Z.to_numpy()
    cm = w @ Zv  # masse des modalités (xQ)
    # ACM = AFC du tableau disjonctif complet avec poids des individus
    P = Zv * w[:, None] / Q
    r = w
    c = cm / Q
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    vp = s ** 2
    J = Zv.shape[1]
    n_vp = J - Q
    vp, U, s, Vt = vp[:n_vp], U[:, :n_vp], s[:n_vp], Vt[:n_vp]
    k = min(n_axes, len(vp))
    axes = _noms_axes(k)
    F = U[:, :k] * s[:k] / np.sqrt(r)[:, None]
    G = Vt[:k].T * s[:k] / np.sqrt(c)[:, None]
    d2i = np.sum((Zv / Q / c - 1) ** 2 * c, axis=1)
    d2m = 1 / cm - 1
    ind = {"coord": pd.DataFrame(F, index=df.index, columns=axes),
           "contrib": pd.DataFrame(r[:, None] * F ** 2 / vp[:k] * 100, index=df.index, columns=axes),
           "cos2": pd.DataFrame(F ** 2 / d2i[:, None], index=df.index, columns=axes)}
    n = len(df)
    mods = {"coord": pd.DataFrame(G, index=Z.columns, columns=axes),
            "contrib": pd.DataFrame(c[:, None] * G ** 2 / vp[:k] * 100, index=Z.columns, columns=axes),
            "cos2": pd.DataFrame(G ** 2 / d2m[:, None], index=Z.columns, columns=axes),
            "v_test": pd.DataFrame(G * np.sqrt((n - 1) * cm / (1 - cm))[:, None], index=Z.columns, columns=axes)}
    # rapport de corrélation (eta²) des variables actives avec les axes
    eta2 = {}
    for v in variables:
        somme = np.zeros(k)
        for m in d[v].unique():
            sel = (d[v] == m).to_numpy()
            somme += w[sel].sum() * np.average(F[sel], axis=0, weights=w[sel]) ** 2
        eta2[v] = somme / vp[:k]
    mods["eta2"] = pd.DataFrame(eta2, index=axes).T
    vpt = _tableau_vp(vp)
    if correction_benzecri:
        seuil = 1 / Q
        corr = np.where(vp > seuil, (Q / (Q - 1) * (vp - seuil)) ** 2, 0.0)
        vpt["valeur_propre_benzecri"] = corr
        vpt["pct_benzecri"] = corr / corr.sum() * 100 if corr.sum() > 0 else 0.0
    supp = {}
    if sup:
        Zs = pd.get_dummies(d[sup], prefix_sep="=").astype(float)
        coord = []
        for col in Zs.columns:
            m = Zs[col].to_numpy() == 1
            coord.append(np.average(F[m], axis=0, weights=w[m]) / np.sqrt(vp[:k]))
        cms = w @ Zs.to_numpy()
        coord = np.asarray(coord)
        supp["modalites"] = {
            "coord": pd.DataFrame(coord, index=Zs.columns, columns=axes),
            "v_test": pd.DataFrame(coord * np.sqrt((n - 1) * cms / (1 - cms))[:, None], index=Zs.columns, columns=axes),
        }
    return ResultatFactoriel("ACM", vpt, ind, mods, supp, {"Q": Q, "J": J, "masses": cm})


def classification(
    coordonnees: pd.DataFrame, n_classes: int | None = None, methode: str = "ward",
    max_classes: int = 10, poids=None,
) -> dict:
    """Classification ascendante hiérarchique (CAH) sur des coordonnées factorielles.

    Parameters
    ----------
    coordonnees : DataFrame
        Coordonnées des individus (ex. ``res.individus["coord"].iloc[:, :5]``)
        ou toute matrice numérique standardisée.
    n_classes : int, optional
        Nombre de classes. Si ``None``, choisi au plus grand saut relatif
        d'inertie (entre 3 et ``max_classes``).
    methode : str
        Critère d'agrégation (``"ward"`` recommandé).

    Returns
    -------
    dict : ``classes`` (Series), ``n_classes``, ``sauts`` (gain d'inertie),
    ``arbre`` (matrice de liaison scipy, pour un dendrogramme).
    """
    X = coordonnees.to_numpy(dtype=float)
    if poids is not None:
        raise NotImplementedError("CAH pondérée non disponible : dupliquer ou utiliser k-means pondéré.")
    L = hierarchy.linkage(X, method=methode)
    hauteurs = L[::-1, 2]
    sauts = pd.DataFrame({"n_classes": np.arange(2, min(max_classes, len(hauteurs)) + 1),
                          "hauteur": hauteurs[: min(max_classes, len(hauteurs)) - 1]})
    if n_classes is None:
        # hauteurs[k-2] = coût de la fusion qui ferait passer de k à k-1 classes :
        # on coupe là où cette fusion est coûteuse comparée à la suivante (k+1 -> k)
        candidats = [k for k in range(3, max_classes + 1) if k - 1 < len(hauteurs)]
        if candidats:
            n_classes = int(max(candidats, key=lambda k: hauteurs[k - 2] / hauteurs[k - 1]))
        else:
            n_classes = 2
    cl = hierarchy.fcluster(L, n_classes, criterion="maxclust")
    return {"classes": pd.Series(cl, index=coordonnees.index, name="classe"), "n_classes": n_classes,
            "sauts": sauts, "arbre": L}


def caracteriser_classes(
    df: pd.DataFrame, classes: pd.Series, variables: Sequence[str], seuil_p: float = 0.05,
) -> pd.DataFrame:
    """Modalités sur- et sous-représentées dans chaque classe (test hypergéométrique, v-test).

    Équivalent de ``catdes`` (FactoMineR) pour les variables qualitatives.

    Returns
    -------
    DataFrame avec les colonnes ``classe``, ``variable``, ``modalite``,
    ``pct_dans_classe``, ``pct_global``, ``pct_classe_dans_modalite``, ``v_test``,
    ``p_valeur``, triées par v-test décroissant.
    """
    variables = en_liste(variables)
    d = df.loc[classes.index, variables].astype(object).where(df.loc[classes.index, variables].notna(), "NA")
    N = len(d)
    lignes = []
    for k in sorted(classes.unique()):
        dans = (classes == k).to_numpy()
        nk = dans.sum()
        for v in variables:
            for m in d[v].unique():
                am = (d[v] == m).to_numpy()
                nm = am.sum()
                nkm = (am & dans).sum()
                # p-valeur hypergéométrique bilatérale (approche FactoMineR)
                if nkm >= nk * nm / N:
                    p = stats.hypergeom.sf(nkm - 1, N, nm, nk)
                    signe = 1
                else:
                    p = stats.hypergeom.cdf(nkm, N, nm, nk)
                    signe = -1
                p2 = min(1.0, 2 * p)
                vt = signe * stats.norm.isf(max(p2 / 2, 1e-300))
                lignes.append({"classe": k, "variable": v, "modalite": m, "effectif": int(nkm),
                               "pct_dans_classe": nkm / nk * 100, "pct_global": nm / N * 100,
                               "pct_classe_dans_modalite": nkm / nm * 100 if nm else np.nan,
                               "v_test": vt, "p_valeur": p2})
    out = pd.DataFrame(lignes)
    out = out[out["p_valeur"] < seuil_p]
    return out.sort_values(["classe", "v_test"], ascending=[True, False]).reset_index(drop=True)
