"""Estimation d'effets causaux sur données observationnelles.

Quand on ne peut pas tirer au sort qui bénéficie d'un dispositif (CEJ, PMSMP,
formation…), comparer bénéficiaires et non-bénéficiaires mélange l'effet du
dispositif et les différences de profil. Ce module propose les méthodes
usuelles pour « rendre comparables » les groupes sur les caractéristiques
observées :

* score de propension et diagnostic d'**équilibre** des covariables ;
* **appariement** sur le score de propension ;
* **pondération inverse** par le score de propension (IPW) ;
* estimateur **doublement robuste** (AIPW) ;
* **différence de différences** (avant/après x traités/témoins) ;
* **E-value** : sensibilité à un facteur de confusion non observé.

.. warning::
   Ces méthodes ne corrigent que les biais liés aux variables **observées et
   incluses**. L'hypothèse d'absence de facteur de confusion non mesuré
   (motivation, réseau…) n'est pas testable : discuter les résultats avec
   prudence et compléter par une analyse de sensibilité (:func:`e_value`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import regression as _reg
from ._encodage import Encodeur
from ._glm import ajuster_glm
from ._utils import en_liste, verifier_colonnes

__all__ = [
    "score_propension",
    "equilibre_covariables",
    "poids_ipw",
    "appariement",
    "estimer_effet",
    "ResultatEffet",
    "difference_en_differences",
    "e_value",
]


def _binaire(s: pd.Series, nom: str) -> np.ndarray:
    v = set(pd.unique(s.dropna()).tolist())
    if not v <= {0, 1, True, False}:
        raise ValueError(f"`{nom}` doit être codé 0/1 ou booléen (valeurs : {sorted(map(str, v))}).")
    return s.astype(float).to_numpy()


def score_propension(
    df: pd.DataFrame, traitement: str, covariables: Sequence[str],
    reference: Mapping[str, object] | None = None, interactions=None,
) -> pd.Series:
    """Probabilité estimée de recevoir le traitement compte tenu des covariables (logistique).

    Returns
    -------
    Series (index de ``df``, NaN pour les lignes incomplètes). Le modèle est
    disponible dans ``serie.attrs["modele"]``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mod = _reg.regression_logistique(df, traitement, covariables, reference=reference,
                                         interactions=interactions)
    ps = pd.Series(np.nan, index=df.index, name="score_propension")
    ps.loc[mod.X.index] = mod.ajustement.mu
    ps.attrs["modele"] = mod
    return ps


def equilibre_covariables(
    df: pd.DataFrame, traitement: str, covariables: Sequence[str], poids: str | np.ndarray | None = None,
    seuil: float = 0.1,
) -> pd.DataFrame:
    """Différences standardisées (SMD) entre traités et témoins, pondérées ou non.

    Règle usuelle : une valeur absolue de SMD inférieure à 0,1 indique un bon équilibre. À calculer avant et
    après appariement/pondération pour vérifier que les groupes sont comparables.
    """
    covariables = en_liste(covariables)
    verifier_colonnes(df, [traitement, *covariables])
    d = df[[traitement, *covariables]].dropna()
    t = _binaire(d[traitement], "traitement")
    if poids is None:
        w = np.ones(len(d))
    elif isinstance(poids, str):
        w = df.loc[d.index, poids].to_numpy(dtype=float)
    else:
        w = pd.Series(np.asarray(poids, dtype=float), index=df.index).loc[d.index].to_numpy()
    X = Encodeur(covariables, constante=False).fit_transform(d)
    lignes = []
    for c in X.columns:
        x = X[c].to_numpy(dtype=float)
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        # écart-type groupé non pondéré (convention : dénominateur fixe avant/après)
        s = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
        smd = (m1 - m0) / s if s > 0 else 0.0
        lignes.append({"covariable": c, "moyenne_traites": m1, "moyenne_temoins": m0, "smd": smd,
                       "equilibre": abs(smd) < seuil})
    return pd.DataFrame(lignes)


def poids_ipw(ps, traitement, estimand: str = "ATE", tronquer: tuple[float, float] | None = (0.01, 0.99),
              stabilises: bool = True) -> np.ndarray:
    """Poids de pondération inverse par le score de propension.

    Parameters
    ----------
    ps : array
        Scores de propension.
    traitement : array
        Indicateur 0/1.
    estimand : {"ATE", "ATT"}
        Effet moyen dans la population (ATE) ou chez les traités (ATT).
    tronquer : (float, float) or None
        Bornes de troncature des scores (évite les poids extrêmes).
    stabilises : bool
        Poids stabilisés (multipliés par la part de traités/témoins) pour l'ATE.
    """
    e = np.asarray(ps, dtype=float)
    t = np.asarray(traitement, dtype=float)
    if tronquer is not None:
        e = np.clip(e, *tronquer)
    if estimand == "ATE":
        w = t / e + (1 - t) / (1 - e)
        if stabilises:
            pt = t.mean()
            w = np.where(t == 1, pt / e, (1 - pt) / (1 - e))
    elif estimand == "ATT":
        w = np.where(t == 1, 1.0, e / (1 - e))
    else:
        raise ValueError("estimand : 'ATE' ou 'ATT'.")
    return w


def appariement(
    df: pd.DataFrame, traitement: str, covariables: Sequence[str] | None = None, score: str | None = None,
    ratio: int = 1, caliper: float | None = 0.2, remise: bool = False, graine: int = 0,
) -> pd.DataFrame:
    """Appariement au plus proche voisin sur le logit du score de propension.

    Parameters
    ----------
    covariables : list of str, optional
        Covariables du score de propension (ignoré si ``score`` est fourni).
    score : str, optional
        Colonne contenant un score de propension déjà calculé.
    ratio : int
        Nombre de témoins par traité.
    caliper : float or None
        Distance maximale, en écarts-types du logit du score (0,2 recommandé).
        Les traités sans témoin assez proche sont écartés.
    remise : bool
        Appariement avec remise (un témoin peut servir plusieurs fois).

    Returns
    -------
    DataFrame
        Échantillon apparié avec ``_poids_appariement`` (1 pour les traités,
        1/ratio cumulé pour les témoins) et ``_groupe_appariement``. L'effet
        estimé sur cet échantillon est un ATT.
    """
    verifier_colonnes(df, [traitement, score] if score else [traitement, *en_liste(covariables)])
    if score is None:
        ps = score_propension(df, traitement, covariables)
    else:
        ps = df[score].astype(float)
    d = df.loc[ps.notna()].copy()
    e = np.clip(ps.loc[d.index].to_numpy(), 1e-6, 1 - 1e-6)
    lg = np.log(e / (1 - e))
    t = _binaire(d[traitement], "traitement")
    tol = caliper * lg.std() if caliper is not None else np.inf
    it = np.flatnonzero(t == 1)
    ic = np.flatnonzero(t == 0)
    rng = np.random.default_rng(graine)
    rng.shuffle(it)
    ordre_c = ic[np.argsort(lg[ic])]
    lg_c = lg[ordre_c]
    dispo = np.ones(len(ordre_c), dtype=bool)
    poids = np.zeros(len(d))
    groupe = np.full(len(d), -1)
    n_non_apparies = 0
    for g, i in enumerate(it):
        choisis = []
        pos = np.searchsorted(lg_c, lg[i])
        gauche, droite = pos - 1, pos
        while len(choisis) < ratio:
            if not remise:
                while gauche >= 0 and not dispo[gauche]:
                    gauche -= 1
                while droite < len(lg_c) and not dispo[droite]:
                    droite += 1
            dg = abs(lg[i] - lg_c[gauche]) if gauche >= 0 else np.inf
            dd = abs(lg_c[droite] - lg[i]) if droite < len(lg_c) else np.inf
            if min(dg, dd) > tol:
                break
            if dg <= dd:
                choisis.append(gauche)
                gauche -= 1
            else:
                choisis.append(droite)
                droite += 1
        if not choisis:
            n_non_apparies += 1
            continue
        poids[i] += 1.0
        groupe[i] = g
        for c in choisis:
            j = ordre_c[c]
            poids[j] += 1.0 / len(choisis)
            groupe[j] = g if groupe[j] == -1 else groupe[j]
            if not remise:
                dispo[c] = False
    if n_non_apparies:
        warnings.warn(f"{n_non_apparies} traité(s) sans témoin dans le caliper ont été écartés.", stacklevel=2)
    d["_poids_appariement"] = poids
    d["_groupe_appariement"] = groupe
    d["_score_propension"] = e
    return d.loc[poids > 0]


@dataclass
class ResultatEffet:
    """Résultat d'une estimation d'effet de traitement.

    ``effet`` est une différence moyenne (en points de probabilité pour un
    résultat binaire : x100 = points de pourcentage).
    """

    methode: str
    estimand: str
    effet: float
    erreur_std: float
    ic_bas: float
    ic_haut: float
    p_valeur: float
    n_traites: int
    n_temoins: int
    equilibre_avant: pd.DataFrame | None = field(default=None, repr=False)
    equilibre_apres: pd.DataFrame | None = field(default=None, repr=False)
    details: dict = field(default_factory=dict, repr=False)

    def resume(self) -> str:
        """Résumé texte."""
        lignes = [f"Effet du traitement - méthode : {self.methode} ({self.estimand})",
                  f"Effet estimé : {self.effet:.4f}  (e.-t. {self.erreur_std:.4f})",
                  f"IC : [{self.ic_bas:.4f} ; {self.ic_haut:.4f}]   p = {self.p_valeur:.4g}",
                  f"Traités : {self.n_traites} | témoins : {self.n_temoins}"]
        if self.equilibre_apres is not None:
            lignes.append(f"|SMD| max après ajustement : {self.equilibre_apres['smd'].abs().max():.3f}")
        return "\n".join(lignes)

    def __str__(self) -> str:  # pragma: no cover
        return self.resume()


def _wls_effet(y, t, w, X=None):
    Z = np.column_stack([np.ones(len(y)), t] + ([X] if X is not None else []))
    aj = ajuster_glm(Z, y, "gaussienne", poids=w, type_cov="HC1")
    return float(aj.coef[1]), float(np.sqrt(aj.cov[1, 1]))


def estimer_effet(
    df: pd.DataFrame,
    traitement: str,
    resultat: str,
    covariables: Sequence[str],
    methode: str = "aipw",
    estimand: str = "ATE",
    tronquer: tuple[float, float] | None = (0.01, 0.99),
    caliper: float | None = 0.2,
    bootstrap: int = 0,
    graine: int = 0,
    alpha: float = 0.05,
) -> ResultatEffet:
    """Estime l'effet moyen d'un traitement binaire sur un résultat.

    Parameters
    ----------
    traitement : str
        Indicateur 0/1 du traitement (ex. ``"a_fait_pmsmp"``).
    resultat : str
        Variable de résultat, numérique ou 0/1 (ex. ``"acces_emploi"``).
    covariables : list of str
        Facteurs de confusion observés (caractéristiques **avant** traitement).
    methode : {"aipw", "ipw", "appariement", "regression"}
        * ``"aipw"`` (défaut) : doublement robuste (correct si le modèle de
          propension **ou** le modèle de résultat est bien spécifié) ;
        * ``"ipw"`` : pondération inverse ;
        * ``"appariement"`` : plus proche voisin (estimand ATT) ;
        * ``"regression"`` : ajustement par régression linéaire.
    estimand : {"ATE", "ATT"}
    bootstrap : int
        Nombre de réplications bootstrap pour l'erreur-type (0 = analytique).

    Returns
    -------
    ResultatEffet
    """
    covariables = en_liste(covariables)
    verifier_colonnes(df, [traitement, resultat, *covariables])
    d = df[[traitement, resultat, *covariables]].dropna().copy()
    t = _binaire(d[traitement], "traitement")
    y = d[resultat].astype(float).to_numpy()
    y_bin = set(np.unique(y)) <= {0.0, 1.0}
    if methode == "appariement" and estimand != "ATT":
        warnings.warn("L'appariement estime un ATT : estimand forcé à 'ATT'.", stacklevel=2)
        estimand = "ATT"
    eq_avant = equilibre_covariables(d, traitement, covariables)

    def calcul(dd: pd.DataFrame) -> tuple[float, float, pd.DataFrame | None]:
        tt = dd[traitement].astype(float).to_numpy()
        yy = dd[resultat].astype(float).to_numpy()
        if methode == "regression":
            X = Encodeur(covariables, constante=False).fit_transform(dd).to_numpy(dtype=float)
            if estimand == "ATT":
                # régression sur les témoins puis prédiction contrefactuelle pour les traités
                Z = np.column_stack([np.ones(len(yy)), X])
                b = np.linalg.lstsq(Z[tt == 0], yy[tt == 0], rcond=None)[0]
                eff = float(np.mean(yy[tt == 1] - Z[tt == 1] @ b))
                return eff, np.nan, None
            eff, se = _wls_effet(yy, tt, None, X)
            return eff, se, None
        ps = score_propension(dd, traitement, covariables).to_numpy()
        if methode == "ipw":
            w = poids_ipw(ps, tt, estimand, tronquer)
            eff, se = _wls_effet(yy, tt, w)
            return eff, se, equilibre_covariables(dd, traitement, covariables, poids=w)
        if methode == "appariement":
            dd2 = dd.assign(_ps=ps)
            m = appariement(dd2, traitement, score="_ps", caliper=caliper, graine=graine)
            w = m["_poids_appariement"].to_numpy()
            eff, se = _wls_effet(m[resultat].astype(float).to_numpy(), m[traitement].astype(float).to_numpy(), w)
            return eff, se, equilibre_covariables(m, traitement, covariables, poids=w)
        if methode == "aipw":
            e = np.clip(ps, *(tronquer or (1e-6, 1 - 1e-6)))
            fam = "binomiale" if y_bin else "gaussienne"
            enc = Encodeur(covariables).fit(dd)
            X = enc.transform(dd).to_numpy(dtype=float)
            preds = {}
            for g in (0, 1):
                aj = ajuster_glm(X[tt == g], yy[tt == g], fam)
                eta = X @ aj.coef
                preds[g] = 1 / (1 + np.exp(-eta)) if y_bin else eta
            m0, m1 = preds[0], preds[1]
            if estimand == "ATE":
                psi = m1 - m0 + tt * (yy - m1) / e - (1 - tt) * (yy - m0) / (1 - e)
                eff = float(psi.mean())
                se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
            else:
                pbar = tt.mean()
                r = tt * (yy - m0) - (1 - tt) * e / (1 - e) * (yy - m0)
                eff = float(r.sum() / tt.sum())
                phi = (r - tt * eff) / pbar
                se = float(phi.std(ddof=1) / np.sqrt(len(phi)))
            w = poids_ipw(ps, tt, estimand, tronquer)
            return eff, se, equilibre_covariables(dd, traitement, covariables, poids=w)
        raise ValueError("methode : 'aipw', 'ipw', 'appariement' ou 'regression'.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eff, se, eq_apres = calcul(d)
        details = {}
        if bootstrap or not np.isfinite(se):
            B = bootstrap or 200
            rng = np.random.default_rng(graine)
            reps = []
            for _ in range(B):
                idx = rng.integers(0, len(d), len(d))
                try:
                    reps.append(calcul(d.iloc[idx].reset_index(drop=True))[0])
                except (np.linalg.LinAlgError, ValueError):
                    continue
            reps = np.asarray(reps)
            se = float(np.nanstd(reps, ddof=1))
            details["bootstrap"] = reps
    q = stats.norm.ppf(1 - alpha / 2)
    z = eff / se if se > 0 else np.nan
    details.update({"resultat_binaire": y_bin})
    return ResultatEffet(
        methode=methode, estimand=estimand, effet=eff, erreur_std=se, ic_bas=eff - q * se,
        ic_haut=eff + q * se, p_valeur=float(2 * stats.norm.sf(abs(z))),
        n_traites=int(t.sum()), n_temoins=int((1 - t).sum()), equilibre_avant=eq_avant,
        equilibre_apres=eq_apres, details=details,
    )


def difference_en_differences(
    df: pd.DataFrame, resultat: str, traite: str, apres: str,
    covariables: Sequence[str] | None = None, grappes: str | None = None, alpha: float = 0.05,
) -> dict:
    """Différence de différences : effet = (traités après − avant) − (témoins après − avant).

    Estimée par régression ``resultat ~ traite + apres + traitexapres (+ covariables)``.
    L'hypothèse clé est celle des **tendances parallèles** : sans le traitement,
    les deux groupes auraient évolué de la même façon (à vérifier sur les
    périodes antérieures avec :func:`obskit.descriptif.evolution`).

    Parameters
    ----------
    traite : str
        0/1 : groupe exposé (ex. territoire où le dispositif a été déployé).
    apres : str
        0/1 : période postérieure au déploiement.
    grappes : str, optional
        Erreurs-types robustes par grappe (ex. territoire), recommandées.

    Returns
    -------
    dict avec ``effet``, ``erreur_std``, ``ic_bas``, ``ic_haut``, ``p_valeur``,
    ``moyennes`` (tableau 2x2) et ``modele`` (ResultatRegression).
    """
    covariables = en_liste(covariables)
    verifier_colonnes(df, [resultat, traite, apres, *covariables, grappes])
    d = df.copy()
    d["_traite"] = _binaire(d[traite], "traite")
    d["_apres"] = _binaire(d[apres], "apres")
    d["_did"] = d["_traite"] * d["_apres"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mod = _reg.regression_lineaire(d, resultat, ["_traite", "_apres", "_did", *covariables],
                                       erreurs="HC1" if not grappes else "cluster", grappes=grappes, alpha=alpha)
    ligne = mod.tableau.loc["_did"]
    moy = d.pivot_table(index="_traite", columns="_apres", values=resultat, aggfunc="mean")
    moy.index = ["Témoins", "Traités"][: len(moy.index)] if list(moy.index) == [0.0, 1.0] else moy.index
    moy.columns = ["Avant", "Après"][: len(moy.columns)] if list(moy.columns) == [0.0, 1.0] else moy.columns
    return {"effet": float(ligne["coef"]), "erreur_std": float(ligne["erreur_std"]),
            "ic_bas": float(ligne["ic_bas"]), "ic_haut": float(ligne["ic_haut"]),
            "p_valeur": float(ligne["p_valeur"]), "moyennes": moy, "modele": mod}


def e_value(estimation: float, borne_ic: float | None = None, type_mesure: str = "RR") -> dict:
    """E-value (VanderWeele & Ding, 2017) : robustesse à un facteur de confusion non mesuré.

    Force minimale (en rapport de risques) que devrait avoir un facteur de
    confusion non observé, associé à la fois au traitement et au résultat, pour
    annuler l'effet estimé. Plus elle est élevée, plus le résultat est robuste.

    Parameters
    ----------
    estimation : float
        Rapport de risques (RR), odds ratio (``type_mesure="OR"``, résultat
        fréquent : approximation par racine carrée) ou rapport de risques
        instantanés (``"HR"``).
    borne_ic : float, optional
        Borne de l'IC la plus proche de 1 : donne l'E-value de l'IC.
    """
    def conv(x):
        if type_mesure == "OR":
            return np.sqrt(x)
        if type_mesure == "HR":
            return (1 - 0.5 ** np.sqrt(x)) / (1 - 0.5 ** np.sqrt(1 / x))
        return x

    def ev(rr):
        rr = 1 / rr if rr < 1 else rr
        return float(rr + np.sqrt(rr * (rr - 1)))

    out = {"e_value": ev(conv(estimation))}
    if borne_ic is not None:
        b = conv(borne_ic)
        out["e_value_ic"] = 1.0 if (estimation >= 1 and b <= 1) or (estimation < 1 and b >= 1) else ev(b)
    return out
