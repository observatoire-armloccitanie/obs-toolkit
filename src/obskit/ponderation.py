"""Pondération et redressement d'échantillons (enquêtes auprès des jeunes, etc.)."""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np
import pandas as pd

from ._utils import verifier_colonnes

__all__ = ["redressement_raking", "effet_plan", "comparer_marges"]


def redressement_raking(
    df: pd.DataFrame,
    marges: Mapping[str, Mapping[object, float]],
    poids_initiaux: str | None = None,
    total: float | None = None,
    bornes: tuple[float, float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> pd.Series:
    """Calage des poids sur des marges connues par la méthode du *raking* (ratio itératif).

    Typiquement : redresser les répondants d'une enquête pour qu'ils
    reproduisent la structure (sexe, âge, département…) de l'ensemble des
    jeunes accompagnés, connue par i-milo.

    Parameters
    ----------
    marges : dict
        ``{variable: {modalite: cible}}``. Les cibles peuvent être des
        proportions (somme = 1) ou des effectifs ; elles sont normalisées.
    poids_initiaux : str, optional
        Colonne de poids de départ (défaut : 1).
    total : float, optional
        Total des poids finaux (défaut : effectif de l'échantillon ; utiliser
        l'effectif de la population pour obtenir des poids d'extrapolation).
    bornes : (float, float), optional
        Bornes des rapports poids final / poids initial (ex. ``(0.3, 3)``) pour
        éviter des poids extrêmes (raking borné).

    Returns
    -------
    Series des poids calés (même index que ``df``).
    """
    verifier_colonnes(df, list(marges) + ([poids_initiaux] if poids_initiaux else []))
    d0 = np.ones(len(df)) if poids_initiaux is None else df[poids_initiaux].to_numpy(dtype=float)
    total = float(total if total is not None else len(df))
    cibles = {}
    for var, m in marges.items():
        s = pd.Series(m, dtype=float)
        absentes = set(df[var].dropna().unique()) - set(s.index)
        if absentes:
            raise ValueError(f"Modalités de '{var}' sans marge : {sorted(map(str, absentes))}")
        if df[var].isna().any():
            raise ValueError(f"'{var}' contient des valeurs manquantes : imputez-les avant le calage.")
        cibles[var] = s / s.sum() * total
    w = d0 * total / d0.sum()
    converge = False
    for _ in range(max_iter):
        ecart_max = 0.0
        for var, cible in cibles.items():
            somme = pd.Series(w).groupby(df[var].to_numpy()).sum()
            facteurs = (cible / somme).reindex(df[var].to_numpy()).to_numpy()
            w = w * facteurs
            if bornes is not None:
                ratio = np.clip(w / (d0 * total / d0.sum()), *bornes)
                w = ratio * d0 * total / d0.sum()
            ecart_max = max(ecart_max, float(np.max(np.abs(somme.reindex(cible.index) - cible) / cible)))
        if ecart_max < tol:
            converge = True
            break
    if not converge:
        warnings.warn("Le calage n'a pas convergé (bornes trop étroites ou marges incompatibles).", stacklevel=2)
    return pd.Series(w, index=df.index, name="poids_cales")


def effet_plan(poids) -> dict:
    """Effet de plan de Kish dû à la pondération et effectif équivalent.

    ``deff = n · Σw² / (Σw)²`` ; ``n_effectif = n / deff``. Un deff de 1,5 signifie
    que la pondération « coûte » un tiers de l'échantillon en précision.
    """
    w = np.asarray(poids, dtype=float)
    n = len(w)
    deff = n * np.sum(w ** 2) / np.sum(w) ** 2
    return {"n": n, "deff_kish": float(deff), "n_effectif": float(n / deff),
            "poids_min": float(w.min()), "poids_max": float(w.max()),
            "rapport_max_min": float(w.max() / w.min()) if w.min() > 0 else np.inf}


def comparer_marges(df: pd.DataFrame, marges: Mapping[str, Mapping[object, float]],
                    poids: str | None = None) -> pd.DataFrame:
    """Compare la structure de l'échantillon (pondéré ou non) aux marges cibles (en %)."""
    lignes = []
    w = np.ones(len(df)) if poids is None else df[poids].to_numpy(dtype=float)
    for var, m in marges.items():
        s = pd.Series(m, dtype=float)
        cible = s / s.sum() * 100
        obs = pd.Series(w).groupby(df[var].to_numpy()).sum()
        obs = obs / obs.sum() * 100
        for mod in cible.index:
            lignes.append({"variable": var, "modalite": mod, "pct_echantillon": float(obs.get(mod, 0.0)),
                           "pct_cible": float(cible[mod]), "ecart_points": float(obs.get(mod, 0.0) - cible[mod])})
    return pd.DataFrame(lignes)
