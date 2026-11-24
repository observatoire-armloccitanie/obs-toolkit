"""Moteur d'estimation des modèles linéaires généralisés (IRLS).

Implémentation autonome (numpy/scipy) des familles gaussienne (lien identité),
binomiale (lien logit) et de Poisson (lien log), avec erreurs-types classiques,
robustes (HC0, HC1, HC3) ou robustes par grappes (*cluster*).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

FAMILLES = ("gaussienne", "binomiale", "poisson")


@dataclass
class AjustementGLM:
    """Résultats bruts d'un ajustement IRLS (usage interne)."""

    famille: str
    coef: np.ndarray
    cov: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    loglik: float
    deviance: float
    n: int
    p: int
    ddl_resid: float
    dispersion: float
    converge: bool
    iterations: int
    poids_travail: np.ndarray = field(repr=False)
    type_cov: str = "classique"


def _moyenne_et_derivee(famille: str, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renvoie (mu, dmu/deta, variance(mu))."""
    if famille == "gaussienne":
        mu = eta
        return mu, np.ones_like(eta), np.ones_like(eta)
    if famille == "binomiale":
        mu = special.expit(eta)
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        d = mu * (1 - mu)
        return mu, d, d
    if famille == "poisson":
        mu = np.exp(np.clip(eta, -700, 700))
        mu = np.maximum(mu, 1e-300)
        return mu, mu, mu
    raise ValueError(f"Famille inconnue : {famille}. Choix : {FAMILLES}")


def _deviance(famille: str, y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    if famille == "gaussienne":
        return float(np.sum(w * (y - mu) ** 2))
    if famille == "binomiale":
        return float(-2 * np.sum(w * (special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu))))
    ratio = np.where(y > 0, y / mu, 1.0)
    return float(2 * np.sum(w * (special.xlogy(y, ratio) - (y - mu))))


def _loglik(famille: str, y, mu, w, dispersion_mle: float) -> float:
    if famille == "gaussienne":
        wpos = w > 0
        return float(-0.5 * np.sum(
            np.log(2 * np.pi * dispersion_mle / w[wpos]) + w[wpos] * (y[wpos] - mu[wpos]) ** 2 / dispersion_mle
        ))
    if famille == "binomiale":
        return float(np.sum(w * (special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu))))
    return float(np.sum(w * (special.xlogy(y, mu) - mu - special.gammaln(y + 1))))


def ajuster_glm(
    X: np.ndarray,
    y: np.ndarray,
    famille: str = "gaussienne",
    poids: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    type_cov: str = "classique",
    grappes: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> AjustementGLM:
    """Ajuste un GLM par moindres carrés repondérés itératifs.

    Parameters
    ----------
    X : ndarray (n, p)
        Matrice de plan (constante incluse si souhaitée).
    y : ndarray (n,)
        Variable réponse (0/1 pour la famille binomiale, comptages pour Poisson).
    famille : {"gaussienne", "binomiale", "poisson"}
    poids : ndarray, optional
        Poids d'observation (pondération d'enquête ou fréquences).
    offset : ndarray, optional
        Terme fixe ajouté au prédicteur linéaire (ex. ``log(exposition)``).
    type_cov : {"classique", "HC0", "HC1", "HC3", "cluster"}
        Type d'erreurs-types. ``"cluster"`` exige ``grappes``.
    grappes : ndarray, optional
        Identifiant de grappe (ex. Mission Locale) pour des erreurs robustes par grappe.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    w = np.ones(n) if poids is None else np.asarray(poids, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    if famille not in FAMILLES:
        raise ValueError(f"Famille inconnue : {famille}. Choix : {FAMILLES}")
    if famille == "binomiale" and not np.all((y == 0) | (y == 1)):
        raise ValueError("La famille binomiale exige une réponse codée 0/1.")
    if famille == "poisson" and np.any(y < 0):
        raise ValueError("La famille de Poisson exige une réponse positive ou nulle.")
    if np.linalg.matrix_rank(X * np.sqrt(w)[:, None]) < p:
        raise np.linalg.LinAlgError(
            "Matrice de plan non de plein rang : variables colinéaires ou modalité vide. "
            "Vérifiez les explicatives (doublons, modalités sans effectif, VIF)."
        )

    # initialisation
    if famille == "gaussienne":
        eta = np.zeros(n)
    elif famille == "binomiale":
        mu0 = (w * y + 0.5) / (w + 1)
        eta = special.logit(mu0)
    else:
        eta = np.log(y + 0.1)
    coef = np.zeros(p)
    dev_prec = np.inf
    converge = False
    it = 0
    for it in range(1, max_iter + 1):
        mu, dmu, var = _moyenne_et_derivee(famille, eta)
        z = (eta - off) + (y - mu) / dmu
        W = w * dmu ** 2 / var
        sw = np.sqrt(W)
        coef_new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
        eta_new = X @ coef_new + off
        mu_new, _, _ = _moyenne_et_derivee(famille, eta_new)
        dev = _deviance(famille, y, mu_new, w)
        # demi-pas si la déviance augmente (robustesse)
        k = 0
        while np.isfinite(dev_prec) and dev > dev_prec * (1 + 1e-12) and k < 30:
            coef_new = (coef_new + coef) / 2
            eta_new = X @ coef_new + off
            mu_new, _, _ = _moyenne_et_derivee(famille, eta_new)
            dev = _deviance(famille, y, mu_new, w)
            k += 1
        coef, eta = coef_new, eta_new
        if famille == "gaussienne" or abs(dev - dev_prec) / (abs(dev) + 0.1) < tol:
            converge = True
            break
        dev_prec = dev

    mu, dmu, var = _moyenne_et_derivee(famille, eta)
    W = w * dmu ** 2 / var
    XtWX = X.T @ (X * W[:, None])
    pain = np.linalg.pinv(XtWX)
    ddl = n - p
    if famille == "gaussienne":
        dispersion = float(np.sum(w * (y - mu) ** 2) / ddl) if ddl > 0 else np.nan
        disp_mle = float(np.sum(w * (y - mu) ** 2) / n)
    else:
        dispersion = 1.0
        disp_mle = 1.0
    if famille == "binomiale" and (np.max(np.abs(coef)) > 15 or not converge):
        warnings.warn(
            "Coefficients très élevés ou non-convergence : possible séparation (quasi-)complète "
            "(une modalité prédit parfaitement la cible). Regroupez des modalités.",
            stacklevel=3,
        )

    correspondance = {"classique": "classique", "cluster": "cluster",
                      "hc0": "HC0", "hc1": "HC1", "hc3": "HC3"}
    type_cov_norm = correspondance.get(str(type_cov).lower(), str(type_cov))
    if type_cov_norm == "classique":
        cov = dispersion * pain
    else:
        # contributions au score
        u = X * (w * (y - mu) * dmu / var)[:, None]
        if type_cov_norm == "cluster":
            if grappes is None:
                raise ValueError("type_cov='cluster' exige l'argument `grappes`.")
            codes, idx = np.unique(np.asarray(grappes), return_inverse=True)
            G = len(codes)
            ug = np.zeros((G, p))
            np.add.at(ug, idx, u)
            viande = ug.T @ ug
            corr = G / (G - 1) * (n - 1) / (n - p)
            cov = corr * pain @ viande @ pain
        elif type_cov_norm in ("HC0", "HC1", "HC3"):
            if type_cov_norm == "HC3":
                h = np.einsum("ij,jk,ik->i", X, pain, X) * W
                u = u / (1 - np.clip(h, 0, 1 - 1e-12))[:, None]
            viande = u.T @ u
            cov = pain @ viande @ pain
            if type_cov_norm == "HC1":
                cov *= n / (n - p)
        else:
            raise ValueError("type_cov doit valoir 'classique', 'HC0', 'HC1', 'HC3' ou 'cluster'.")

    return AjustementGLM(
        famille=famille, coef=coef, cov=cov, mu=mu, eta=eta,
        loglik=_loglik(famille, y, mu, w, disp_mle), deviance=_deviance(famille, y, mu, w),
        n=n, p=p, ddl_resid=ddl, dispersion=dispersion, converge=converge,
        iterations=it, poids_travail=W, type_cov=type_cov_norm,
    )


def tableau_coefficients(aj: AjustementGLM, noms: list[str], alpha: float = 0.05):
    """Tableau des coefficients : estimation, erreur-type, statistique, p-valeur, IC."""
    import pandas as pd

    se = np.sqrt(np.clip(np.diag(aj.cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = aj.coef / se
    if aj.famille == "gaussienne" and aj.type_cov == "classique":
        loi = stats.t(df=aj.ddl_resid)
    else:
        loi = stats.norm()
    p = 2 * loi.sf(np.abs(stat))
    q = loi.ppf(1 - alpha / 2)
    tab = pd.DataFrame(
        {
            "coef": aj.coef,
            "erreur_std": se,
            "stat": stat,
            "p_valeur": p,
            "ic_bas": aj.coef - q * se,
            "ic_haut": aj.coef + q * se,
        },
        index=pd.Index(noms, name="terme"),
    )
    return tab
