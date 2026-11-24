"""Statistiques descriptives : tris à plat, tris croisés, « tableau 1 », évolutions.

Toutes les fonctions acceptent une colonne de pondération (``poids=``) et
renvoient des DataFrames prêts à être exportés (voir :func:`obskit.io.exporter_excel`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import inference
from ._utils import en_liste, est_categorielle, formater_p, poids_ou_uns, verifier_colonnes

__all__ = [
    "resume_variables",
    "valeurs_manquantes",
    "statistiques_descriptives",
    "tableau_frequences",
    "tableau_croise",
    "tableau_un",
    "evolution",
    "quantiles_ponderes",
]


def resume_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Panorama de toutes les colonnes : type, effectif, manquants, modalités, exemples.

    Première fonction à lancer sur une nouvelle extraction.
    """
    lignes = []
    for c in df.columns:
        s = df[c]
        ligne = {"variable": c, "type": str(s.dtype), "n_valides": int(s.notna().sum()),
                 "n_manquants": int(s.isna().sum()), "pct_manquants": float(s.isna().mean() * 100),
                 "n_modalites": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            ligne.update({"moyenne": s.mean(), "mediane": s.median(), "min": s.min(), "max": s.max()})
        elif pd.api.types.is_datetime64_any_dtype(s):
            ligne.update({"min": s.min(), "max": s.max()})
        else:
            vc = s.value_counts()
            if len(vc):
                ligne.update({"modalite_principale": vc.index[0],
                              "pct_modalite_principale": vc.iloc[0] / s.notna().sum() * 100})
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def valeurs_manquantes(df: pd.DataFrame, seuil: float = 0.0) -> pd.DataFrame:
    """Effectif et part de valeurs manquantes par colonne (triées), au-delà de ``seuil`` %."""
    n = df.isna().sum()
    out = pd.DataFrame({"n_manquants": n, "pct_manquants": n / len(df) * 100})
    out = out[out["pct_manquants"] > seuil].sort_values("pct_manquants", ascending=False)
    out.index.name = "variable"
    return out


def quantiles_ponderes(x, poids, q: Sequence[float]) -> np.ndarray:
    """Quantiles pondérés (interpolation sur la fonction de répartition pondérée)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(poids, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(q, cw, x)


def statistiques_descriptives(
    df: pd.DataFrame,
    variables: Sequence[str] | None = None,
    par: str | Sequence[str] | None = None,
    poids: str | None = None,
    decimales: int | None = None,
) -> pd.DataFrame:
    """Indicateurs de position et de dispersion des variables numériques.

    Parameters
    ----------
    variables : list of str, optional
        Variables numériques (défaut : toutes les numériques).
    par : str or list, optional
        Variable(s) de ventilation (ex. ``"departement"``).
    poids : str, optional
        Pondération (moyenne, écart-type et quantiles pondérés).
    decimales : int, optional
        Arrondi du résultat.

    Returns
    -------
    DataFrame : ``n, moyenne, ecart_type, min, q1, mediane, q3, max, n_manquants``.
    """
    par = en_liste(par)
    if variables is None:
        variables = [c for c in df.select_dtypes("number").columns if c not in par and c != poids]
    variables = en_liste(variables)
    verifier_colonnes(df, [*variables, *par, poids])

    def calc(d: pd.DataFrame) -> pd.DataFrame:
        lignes = {}
        for v in variables:
            s = pd.to_numeric(d[v], errors="coerce")
            m = s.notna()
            if poids:
                w = d.loc[m, poids].to_numpy(dtype=float)
                x = s[m].to_numpy(dtype=float)
                moy = np.average(x, weights=w) if w.sum() > 0 else np.nan
                et = np.sqrt(np.average((x - moy) ** 2, weights=w)) if w.sum() > 0 else np.nan
                q = quantiles_ponderes(x, w, [0.25, 0.5, 0.75]) if len(x) else [np.nan] * 3
                n = w.sum()
            else:
                x = s[m]
                moy, et = x.mean(), x.std()
                q = x.quantile([0.25, 0.5, 0.75]).to_numpy() if len(x) else [np.nan] * 3
                n = len(x)
            lignes[v] = {"n": n, "moyenne": moy, "ecart_type": et, "min": s.min(), "q1": q[0],
                         "mediane": q[1], "q3": q[2], "max": s.max(), "n_manquants": int((~m).sum())}
        out = pd.DataFrame(lignes).T
        out.index.name = "variable"
        return out

    if par:
        res = pd.concat({k if isinstance(k, tuple) else (k,): calc(g)
                         for k, g in df.groupby(par, observed=True)}, names=[*par])
    else:
        res = calc(df)
    return res.round(decimales) if decimales is not None else res


def tableau_frequences(
    df: pd.DataFrame,
    variable: str,
    poids: str | None = None,
    tri: str = "effectif",
    inclure_manquants: bool = False,
    ic: bool = True,
    alpha: float = 0.05,
    total: bool = True,
) -> pd.DataFrame:
    """Tri à plat : effectifs, pourcentages, pourcentages cumulés et IC (Wilson).

    Parameters
    ----------
    tri : {"effectif", "modalite", "aucun"}
        Ordre des lignes. Pour une variable ``category`` ordonnée, ``"modalite"``
        respecte l'ordre des catégories.
    inclure_manquants : bool
        Ajoute une ligne « Non renseigné ».
    ic : bool
        Ajoute l'intervalle de confiance du pourcentage.
    total : bool
        Ajoute une ligne « Total ».

    Examples
    --------
    >>> tableau_frequences(df, "niveau_formation")     # doctest: +SKIP
    """
    verifier_colonnes(df, [variable, poids])
    s = df[variable]
    w = pd.Series(poids_ou_uns(df, poids), index=df.index)
    if inclure_manquants:
        s = s.astype(object).where(s.notna(), "Non renseigné")
    m = s.notna()
    eff = w[m].groupby(s[m], observed=True, sort=False).sum()
    if tri == "effectif":
        eff = eff.sort_values(ascending=False)
    elif tri == "modalite":
        if isinstance(df[variable].dtype, pd.CategoricalDtype):
            ordre = [c for c in df[variable].cat.categories if c in eff.index]
            ordre += [i for i in eff.index if i not in ordre]
            eff = eff.reindex(ordre)
        else:
            eff = eff.sort_index(key=lambda idx: idx.map(str))
    n_tot = eff.sum()
    out = pd.DataFrame({"effectif": eff, "pct": eff / n_tot * 100})
    out["pct_cumule"] = out["pct"].cumsum()
    if ic:
        n_reel = int(m.sum())
        bornes = [inference.intervalle_proportion(p / 100 * n_reel, n_reel, alpha) for p in out["pct"]]
        out["ic_bas"] = [b[0] * 100 for b in bornes]
        out["ic_haut"] = [b[1] * 100 for b in bornes]
    out.index.name = variable
    if total:
        ligne = {"effectif": n_tot, "pct": 100.0, "pct_cumule": np.nan}
        if ic:
            ligne.update({"ic_bas": np.nan, "ic_haut": np.nan})
        out.index = out.index.astype(object)
        out.loc["Total"] = ligne
    return out


def tableau_croise(
    df: pd.DataFrame,
    ligne: str,
    colonne: str,
    poids: str | None = None,
    pourcentage: str | None = "ligne",
    marges: bool = True,
    test: bool = True,
    decimales: int | None = 1,
) -> pd.DataFrame:
    """Tri croisé (effectifs ou pourcentages) avec test d'indépendance.

    Parameters
    ----------
    ligne, colonne : str
        Variables en ligne et en colonne.
    pourcentage : {"ligne", "colonne", "total", None}
        Type de pourcentage ; ``None`` pour les effectifs bruts.
    marges : bool
        Ajoute les totaux (« Ensemble »).
    test : bool
        Calcule le test du χ² ; le résultat est stocké dans ``resultat.attrs["test"]``.

    Returns
    -------
    DataFrame
        Le tableau ; ``attrs["effectifs"]`` contient les effectifs et
        ``attrs["test"]`` le test d'indépendance (χ², p-valeur, V de Cramér).
    """
    verifier_colonnes(df, [ligne, colonne, poids])
    d = df[[c for c in (ligne, colonne, poids) if c]].dropna()
    if poids:
        eff = d.pivot_table(index=ligne, columns=colonne, values=poids, aggfunc="sum",
                            fill_value=0, observed=True)
    else:
        eff = pd.crosstab(d[ligne], d[colonne])
    eff_m = eff.copy()
    eff_m["Ensemble"] = eff_m.sum(axis=1)
    eff_m.loc["Ensemble"] = eff_m.sum(axis=0)
    if pourcentage is None:
        out = eff_m.copy()
    elif pourcentage == "ligne":
        out = eff_m.div(eff_m["Ensemble"], axis=0) * 100
    elif pourcentage == "colonne":
        out = eff_m.div(eff_m.loc["Ensemble"], axis=1) * 100
    elif pourcentage == "total":
        out = eff_m / eff_m.loc["Ensemble", "Ensemble"] * 100
    else:
        raise ValueError("pourcentage : 'ligne', 'colonne', 'total' ou None.")
    if not marges:
        out = out.drop(index="Ensemble", columns="Ensemble")
    if decimales is not None and pourcentage is not None:
        out = out.round(decimales)
    out.attrs["effectifs"] = eff_m
    if test:
        try:
            t = inference.test_independance(d, ligne, colonne, poids=poids)
            out.attrs["test"] = {k: t[k] for k in ("test", "statistique", "ddl", "p_valeur", "v_cramer")}
        except ValueError:
            out.attrs["test"] = None
    return out


def tableau_un(
    df: pd.DataFrame,
    groupe: str | None,
    variables: Sequence[str],
    continues: str = "moyenne",
    tests: bool = True,
    correction: str | None = None,
    decimales: int = 1,
) -> pd.DataFrame:
    """« Tableau 1 » : description de variables par groupe avec tests de comparaison.

    Format standard des publications : pour chaque variable catégorielle
    ``n (%)`` par modalité, pour chaque variable continue ``moyenne (écart-type)``
    ou ``médiane [Q1 ; Q3]``, une colonne par groupe, une colonne « Ensemble »
    et la p-valeur du test adapté (χ²/Fisher, t de Welch/ANOVA ou non paramétrique).

    Parameters
    ----------
    groupe : str or None
        Variable de comparaison (ex. ``"dispositif"``) ; ``None`` = description globale.
    variables : list of str
        Variables à décrire.
    continues : {"moyenne", "mediane"}
        Résumé des variables continues.
    correction : {"holm", "bonferroni", "fdr_bh"}, optional
        Correction des p-valeurs pour tests multiples.

    Returns
    -------
    DataFrame indexé par (variable, modalité).
    """
    verifier_colonnes(df, [groupe, *variables])
    groupes = [] if groupe is None else sorted(df[groupe].dropna().unique().tolist(), key=str)
    colonnes = [*groupes, "Ensemble"]
    sous_ensembles = {g: df[df[groupe] == g] for g in groupes}
    sous_ensembles["Ensemble"] = df if groupe is None else df[df[groupe].notna()]
    lignes, index, pvals = [], [], []
    entete = {c: f"n = {len(sous_ensembles[c])}" for c in colonnes}
    lignes.append(entete)
    index.append(("Effectif", ""))
    pvals.append(np.nan)
    for v in variables:
        s = df[v]
        if est_categorielle(s) or s.nunique() <= 2:
            mods = (list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype)
                    else sorted(s.dropna().unique().tolist(), key=str))
            p = np.nan
            if tests and groupe is not None:
                try:
                    p = inference.test_independance(df, v, groupe)["p_valeur"]
                except (ValueError, TypeError):
                    p = np.nan
            for i, m in enumerate(mods):
                ligne = {}
                for c in colonnes:
                    sub = sous_ensembles[c][v]
                    n_m = int((sub == m).sum())
                    tot = int(sub.notna().sum())
                    ligne[c] = f"{n_m} ({n_m / tot * 100:.{decimales}f} %)" if tot else "-"
                lignes.append(ligne)
                index.append((v, str(m)))
                pvals.append(p if i == 0 else np.nan)
        else:
            p = np.nan
            if tests and groupe is not None:
                try:
                    p = inference.comparer_groupes(df, v, groupe, parametrique=(continues == "moyenne"))["p_valeur"]
                except ValueError:
                    p = np.nan
            ligne = {}
            for c in colonnes:
                x = pd.to_numeric(sous_ensembles[c][v], errors="coerce").dropna()
                if continues == "moyenne":
                    ligne[c] = f"{x.mean():.{decimales}f} ({x.std():.{decimales}f})"
                else:
                    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
                    ligne[c] = f"{med:.{decimales}f} [{q1:.{decimales}f} ; {q3:.{decimales}f}]"
            lignes.append(ligne)
            lib = "moyenne (é.-t.)" if continues == "moyenne" else "médiane [Q1 ; Q3]"
            index.append((v, lib))
            pvals.append(p)
        n_na = int(s.isna().sum())
        if n_na:
            lignes.append({c: str(int(sous_ensembles[c][v].isna().sum())) for c in colonnes})
            index.append((v, "Non renseigné"))
            pvals.append(np.nan)
    out = pd.DataFrame(lignes, index=pd.MultiIndex.from_tuples(index, names=["variable", "modalite"]))
    if tests and groupe is not None:
        p = np.asarray(pvals, dtype=float)
        if correction:
            p = inference.corriger_pvaleurs(p, correction)
        out["p_valeur"] = [formater_p(x) if np.isfinite(x) else "" for x in p]
    if groupe is None:
        out = out[["Ensemble"]]
    return out


def evolution(
    df: pd.DataFrame,
    date: str,
    frequence: str = "M",
    par: str | None = None,
    valeur: str | None = None,
    agregation: str = "count",
    cumul: bool = False,
) -> pd.DataFrame:
    """Série temporelle d'un indicateur (effectifs mensuels d'entrées, etc.) et variations.

    Parameters
    ----------
    date : str
        Colonne date.
    frequence : str
        Fréquence pandas : ``"D"``, ``"W"``, ``"M"`` (mois), ``"Q"`` (trimestre), ``"Y"`` (année).
    par : str, optional
        Ventilation (une colonne par modalité).
    valeur : str, optional
        Colonne agrégée (défaut : comptage des lignes).
    agregation : str
        ``"count"``, ``"sum"``, ``"mean"``, ``"median"``…
    cumul : bool
        Renvoie le cumul depuis le début de période.

    Returns
    -------
    DataFrame indexé par période, avec ``variation_pct`` (période précédente) et
    ``glissement_annuel_pct`` (même période N-1) quand ``par`` est absent.
    """
    verifier_colonnes(df, [date, par, valeur])
    d = df.copy()
    d["_periode"] = pd.to_datetime(d[date], errors="coerce").dt.to_period(frequence)
    d = d.dropna(subset=["_periode"])
    cle = valeur or date
    if par:
        out = d.pivot_table(index="_periode", columns=par, values=cle, aggfunc=agregation,
                            fill_value=0, observed=True)
    else:
        out = d.groupby("_periode")[cle].agg(agregation).to_frame("valeur")
    complet = pd.period_range(out.index.min(), out.index.max(), freq=out.index.freq)
    out = out.reindex(complet, fill_value=0)
    out.index.name = "periode"
    if cumul:
        out = out.cumsum()
    if not par:
        out["variation_pct"] = out["valeur"].pct_change(fill_method=None) * 100
        pas = {"M": 12, "Q": 4, "W": 52, "Y": 1, "D": 365}.get(frequence[0].upper(), None)
        if pas:
            out["glissement_annuel_pct"] = out["valeur"].pct_change(pas, fill_method=None) * 100
    return out
