"""Analyse de survie (analyse des durées).

Outils pour étudier la durée jusqu'à un événement (accès à l'emploi, sortie
d'accompagnement, entrée en formation…) en tenant compte des observations
censurées (jeunes encore accompagnés à la date d'extraction).

* :func:`preparer_donnees_survie` : calcule durée et indicateur d'événement à partir de dates ;
* :func:`kaplan_meier` : courbes de survie non paramétriques, médianes, IC ;
* :func:`test_logrank` : comparaison de courbes entre groupes ;
* :func:`modele_cox` : modèle à risques proportionnels de Cox (rapports de risque) ;
* :func:`test_proportionnalite` : vérification de l'hypothèse des risques proportionnels ;
* :func:`incidence_cumulee` : risques concurrents (Aalen-Johansen).

Convention : ``evenement`` vaut 1 si l'événement est observé, 0 si l'observation est censurée.
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._encodage import Encodeur
from ._utils import etoiles, explicatives_par_defaut, formater_p, verifier_colonnes

__all__ = [
    "preparer_donnees_survie",
    "kaplan_meier",
    "ResultatKaplanMeier",
    "test_logrank",
    "modele_cox",
    "ResultatCox",
    "test_proportionnalite",
    "incidence_cumulee",
    "concordance",
]

_UNITES = {"jours": 1.0, "semaines": 7.0, "mois": 365.25 / 12, "annees": 365.25}


# ----------------------------------------------------------------------------
# Préparation
# ----------------------------------------------------------------------------
def preparer_donnees_survie(
    df: pd.DataFrame,
    date_debut: str,
    date_evenement: str,
    date_fin_observation: str | pd.Timestamp,
    unite: str = "mois",
    nom_duree: str = "duree",
    nom_evenement: str = "evenement",
) -> pd.DataFrame:
    """Construit les colonnes durée / événement à partir de dates.

    Parameters
    ----------
    date_debut : str
        Colonne date d'origine (ex. ``"date_entree_cej"``).
    date_evenement : str
        Colonne date de l'événement ; vide (NaT) si l'événement n'a pas eu lieu.
    date_fin_observation : str ou Timestamp
        Date de censure : colonne (date de sortie de suivi propre à chaque
        individu) ou date unique (date d'extraction).
    unite : {"jours", "semaines", "mois", "annees"}

    Returns
    -------
    DataFrame
        Copie de ``df`` avec ``duree`` (≥ 0) et ``evenement`` (0/1). Un
        événement postérieur à la fin d'observation est considéré comme censuré.
    """
    verifier_colonnes(df, [date_debut, date_evenement])
    if unite not in _UNITES:
        raise ValueError(f"unite doit être parmi {list(_UNITES)}")
    out = df.copy()
    debut = pd.to_datetime(out[date_debut], errors="coerce", dayfirst=True)
    evt = pd.to_datetime(out[date_evenement], errors="coerce", dayfirst=True)
    if isinstance(date_fin_observation, str) and date_fin_observation in out.columns:
        fin = pd.to_datetime(out[date_fin_observation], errors="coerce", dayfirst=True)
    else:
        fin = pd.Series(pd.Timestamp(date_fin_observation), index=out.index)
    observe = evt.notna() & (evt <= fin)
    date_sortie = evt.where(observe, fin)
    duree = (date_sortie - debut).dt.days / _UNITES[unite]
    if (duree < 0).any():
        warnings.warn(f"{int((duree < 0).sum())} durée(s) négative(s) mises à NaN (dates incohérentes).",
                      stacklevel=2)
        duree = duree.where(duree >= 0)
    out[nom_duree] = duree
    out[nom_evenement] = observe.astype(int)
    return out


def _extraire(df, duree, evenement, autres=()):
    verifier_colonnes(df, [duree, evenement, *[a for a in autres if a]])
    cols = list(dict.fromkeys([duree, evenement, *[a for a in autres if a]]))
    d = df[cols].dropna()
    n_exclus = len(df) - len(d)
    T = d[duree].to_numpy(dtype=float)
    E = d[evenement].to_numpy(dtype=float)
    if np.any(T < 0):
        raise ValueError("Durées négatives détectées.")
    if not np.all((E == 0) | (E == 1)):
        raise ValueError("`evenement` doit être codé 0 (censure) / 1 (événement).")
    return d, T, E, n_exclus


# ----------------------------------------------------------------------------
# Kaplan-Meier
# ----------------------------------------------------------------------------
def _km_un_groupe(T, E, w, alpha):
    ordre = np.argsort(T, kind="mergesort")
    T, E, w = T[ordre], E[ordre], w[ordre]
    temps = np.unique(T)
    idx = np.searchsorted(T, temps, side="left")
    n_risque = np.cumsum(w[::-1])[::-1][idx]
    d = np.zeros(len(temps))
    c = np.zeros(len(temps))
    pos = np.searchsorted(temps, T)
    np.add.at(d, pos, w * E)
    np.add.at(c, pos, w * (1 - E))
    with np.errstate(divide="ignore", invalid="ignore"):
        facteur = 1 - d / n_risque
        surv = np.cumprod(facteur)
        terme = np.where(n_risque - d > 0, d / (n_risque * (n_risque - d)), np.nan)
        terme = np.where(d == 0, 0.0, terme)
        var_cum = np.cumsum(terme)
        z = stats.norm.ppf(1 - alpha / 2)
        # IC log-log (Kalbfleisch-Prentice), comme R survival (conf.type="log-log") et lifelines
        ll = np.log(-np.log(surv))
        se_ll = np.sqrt(var_cum) / np.abs(np.log(surv))
        ic_bas = np.exp(-np.exp(ll + z * se_ll))
        ic_haut = np.exp(-np.exp(ll - z * se_ll))
    ic_bas = np.where(surv >= 1, 1.0, np.where(surv <= 0, 0.0, ic_bas))
    ic_haut = np.where(surv >= 1, 1.0, np.where(surv <= 0, 0.0, ic_haut))
    tab = pd.DataFrame({"temps": temps, "n_risque": n_risque, "n_evenements": d, "n_censures": c,
                        "survie": surv, "ic_bas": ic_bas, "ic_haut": ic_haut,
                        "erreur_std": surv * np.sqrt(var_cum)})
    t0 = pd.DataFrame({"temps": [0.0], "n_risque": [w.sum()], "n_evenements": [0.0], "n_censures": [0.0],
                       "survie": [1.0], "ic_bas": [1.0], "ic_haut": [1.0], "erreur_std": [0.0]})
    if temps[0] > 0:
        tab = pd.concat([t0, tab], ignore_index=True)
    return tab


def _premier_temps_sous(tab, col, seuil=0.5):
    sel = tab.loc[tab[col] <= seuil, "temps"]
    return float(sel.iloc[0]) if len(sel) else np.nan


@dataclass
class ResultatKaplanMeier:
    """Résultat de :func:`kaplan_meier`.

    Attributes
    ----------
    tables : dict
        Une table de survie par groupe (clé ``"Ensemble"`` sans groupe) avec
        ``temps, n_risque, n_evenements, n_censures, survie, ic_bas, ic_haut``.
    medianes : DataFrame
        Durée médiane (et IC) par groupe : durée au bout de laquelle la moitié
        des individus a connu l'événement.
    logrank : dict or None
        Test du log-rank si un groupe est fourni.
    """

    tables: dict
    medianes: pd.DataFrame
    effectifs: pd.DataFrame
    logrank: dict | None = None
    groupe: str | None = None
    alpha: float = 0.05

    @property
    def table(self) -> pd.DataFrame:
        """Toutes les tables empilées avec une colonne ``groupe``."""
        return pd.concat([t.assign(groupe=g) for g, t in self.tables.items()], ignore_index=True)

    def survie_a(self, temps: float | Sequence[float]) -> pd.DataFrame:
        """Probabilité de « survie » (événement non encore survenu) à des durées données.

        ``1 - survie`` est la proportion cumulée d'individus ayant connu l'événement.
        """
        temps = np.atleast_1d(np.asarray(temps, dtype=float))
        lignes = []
        for g, t in self.tables.items():
            for x in temps:
                i = np.searchsorted(t["temps"].to_numpy(), x, side="right") - 1
                ligne = t.iloc[max(i, 0)]
                lignes.append({"groupe": g, "temps": x, "survie": ligne["survie"],
                               "ic_bas": ligne["ic_bas"], "ic_haut": ligne["ic_haut"],
                               "incidence_cumulee": 1 - ligne["survie"]})
        return pd.DataFrame(lignes)

    def rmst(self, tau: float) -> pd.DataFrame:
        """Durée moyenne restreinte (*restricted mean survival time*) jusqu'à ``tau``.

        Aire sous la courbe de survie entre 0 et ``tau`` : durée moyenne passée
        « sans événement » sur la fenêtre [0, tau]. Utile quand la médiane n'est pas atteinte.
        """
        lignes = []
        for g, t in self.tables.items():
            tt = t["temps"].to_numpy()
            s = t["survie"].to_numpy()
            if tt[0] > 0:
                tt, s = np.r_[0.0, tt], np.r_[1.0, s]
            bornes = np.r_[tt[tt < tau], tau]
            vals = s[: len(bornes) - 1]
            lignes.append({"groupe": g, "tau": tau, "rmst": float(np.sum(np.diff(bornes) * vals))})
        return pd.DataFrame(lignes)

    def tracer(self, ax=None, **kwargs):
        """Trace les courbes (voir :func:`obskit.visualisation.courbes_survie`)."""
        from .visualisation import courbes_survie

        return courbes_survie(self, ax=ax, **kwargs)


def kaplan_meier(
    df: pd.DataFrame,
    duree: str,
    evenement: str,
    groupe: str | None = None,
    poids: str | None = None,
    alpha: float = 0.05,
) -> ResultatKaplanMeier:
    """Estimateur de Kaplan-Meier, globalement ou par groupe.

    Parameters
    ----------
    df : DataFrame
    duree : str
        Colonne de durée (temps jusqu'à l'événement ou la censure).
    evenement : str
        Colonne 0/1 (1 = événement observé).
    groupe : str, optional
        Variable de comparaison (ex. ``"dispositif"``) ; ajoute un test du log-rank.
    poids : str, optional
        Pondération (les IC sont alors approximatifs).
    alpha : float
        Risque des IC (IC log-log).

    Examples
    --------
    >>> km = kaplan_meier(df, "duree", "sortie_emploi", groupe="qpv")   # doctest: +SKIP
    >>> km.medianes                                                    # doctest: +SKIP
    >>> km.survie_a([6, 12])                                           # doctest: +SKIP
    """
    d, T, E, _ = _extraire(df, duree, evenement, [groupe, poids])
    w = d[poids].to_numpy(dtype=float) if poids else np.ones(len(d))
    tables, med, eff = {}, [], []
    if groupe is None:
        groupes = [("Ensemble", np.ones(len(d), dtype=bool))]
    else:
        g = d[groupe]
        mods = sorted(g.unique().tolist(), key=lambda v: (str(type(v)), v))
        groupes = [(m, (g == m).to_numpy()) for m in mods]
    for nom, m in groupes:
        tab = _km_un_groupe(T[m], E[m], w[m], alpha)
        tables[nom] = tab
        med.append({"groupe": nom, "mediane": _premier_temps_sous(tab, "survie"),
                    "ic_bas": _premier_temps_sous(tab, "ic_bas"),
                    "ic_haut": _premier_temps_sous(tab, "ic_haut")})
        eff.append({"groupe": nom, "n": int(m.sum()), "evenements": int(E[m].sum()),
                    "censures": int((1 - E[m]).sum())})
    lr = test_logrank(d, duree, evenement, groupe) if groupe is not None and len(groupes) > 1 else None
    return ResultatKaplanMeier(tables=tables, medianes=pd.DataFrame(med), effectifs=pd.DataFrame(eff),
                               logrank=lr, groupe=groupe, alpha=alpha)


def test_logrank(df: pd.DataFrame, duree: str, evenement: str, groupe: str) -> dict:
    """Test du log-rank (Mantel-Cox) d'égalité des courbes de survie entre k groupes.

    Returns
    -------
    dict
        ``chi2``, ``ddl``, ``p_valeur`` et un tableau ``observes_attendus``.
    """
    d, T, E, _ = _extraire(df, duree, evenement, [groupe])
    g = d[groupe].to_numpy()
    mods = sorted(pd.unique(g).tolist(), key=lambda v: (str(type(v)), v))
    k = len(mods)
    if k < 2:
        raise ValueError("Il faut au moins deux groupes.")
    temps = np.unique(T[E == 1])
    V = np.zeros((k, k))
    Ts = np.sort(T)
    n_tot = len(T) - np.searchsorted(Ts, temps, side="left")
    d_tot = np.bincount(np.searchsorted(temps, T[E == 1]), minlength=len(temps))
    n_g = np.zeros((len(temps), k))
    d_g = np.zeros((len(temps), k))
    for j, m in enumerate(mods):
        Tj = np.sort(T[g == m])
        n_g[:, j] = len(Tj) - np.searchsorted(Tj, temps, side="left")
        ev = T[(g == m) & (E == 1)]
        d_g[:, j] = np.bincount(np.searchsorted(temps, ev), minlength=len(temps))
    O = d_g.sum(axis=0)
    frac = n_g / n_tot[:, None]
    Ex = (d_tot[:, None] * frac).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        facteur = np.where(n_tot > 1, d_tot * (n_tot - d_tot) / (n_tot - 1), 0.0)
    for a in range(k):
        for b in range(k):
            V[a, b] = np.sum(facteur * frac[:, a] * ((1.0 if a == b else 0.0) - frac[:, b]))
    diff = (O - Ex)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    ddl = k - 1
    tab = pd.DataFrame({"groupe": mods, "observes": O, "attendus": Ex, "o_sur_e": O / Ex})
    return {"chi2": chi2, "ddl": ddl, "p_valeur": float(stats.chi2.sf(chi2, ddl)), "observes_attendus": tab}


# ----------------------------------------------------------------------------
# Incidence cumulée (risques concurrents)
# ----------------------------------------------------------------------------
def incidence_cumulee(
    df: pd.DataFrame, duree: str, type_evenement: str, censure: object = 0, groupe: str | None = None,
) -> pd.DataFrame:
    """Fonctions d'incidence cumulée en présence de risques concurrents (Aalen-Johansen).

    Quand plusieurs issues sont possibles (emploi, formation, abandon…), ``1 - KM``
    surestime l'incidence de chaque issue. Cette fonction estime correctement la
    probabilité cumulée de chaque type d'événement au cours du temps.

    Parameters
    ----------
    type_evenement : str
        Colonne contenant le type d'issue ; la valeur ``censure`` désigne les
        observations censurées.
    groupe : str, optional
        Calcul séparé par groupe.

    Returns
    -------
    DataFrame long : ``groupe, temps, issue, incidence_cumulee``.
    """
    verifier_colonnes(df, [duree, type_evenement, groupe])
    d = df[[c for c in (duree, type_evenement, groupe) if c]].dropna(subset=[duree, type_evenement])
    sorties = []
    blocs = [("Ensemble", d)] if groupe is None else list(d.groupby(groupe, observed=True))
    issues = [v for v in pd.unique(d[type_evenement]) if v != censure]
    for nom, b in blocs:
        T = b[duree].to_numpy(dtype=float)
        typ = b[type_evenement].to_numpy()
        temps = np.unique(T)
        Ts = np.sort(T)
        n = len(T) - np.searchsorted(Ts, temps, side="left")
        pos = np.searchsorted(temps, T)
        d_tout = np.bincount(pos[typ != censure], minlength=len(temps)).astype(float)
        s_avant = np.r_[1.0, np.cumprod(1 - d_tout / n)[:-1]]
        for iss in issues:
            dk = np.bincount(pos[typ == iss], minlength=len(temps)).astype(float)
            cif = np.cumsum(s_avant * dk / n)
            sorties.append(pd.DataFrame({"groupe": nom, "temps": temps, "issue": iss, "incidence_cumulee": cif}))
    return pd.concat(sorties, ignore_index=True)


# ----------------------------------------------------------------------------
# Modèle de Cox
# ----------------------------------------------------------------------------
def _cox_vraisemblance(beta, X, T, E, strates, methode):
    """Log-vraisemblance partielle, gradient et hessienne (Efron ou Breslow)."""
    p = X.shape[1]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    for s_idx in strates:
        Xs, Ts, Es = X[s_idx], T[s_idx], E[s_idx]
        eta = np.clip(Xs @ beta, -500, 500)
        r = np.exp(eta)
        ordre = np.argsort(-Ts, kind="mergesort")
        Xs, Ts, Es, eta, r = Xs[ordre], Ts[ordre], Es[ordre], eta[ordre], r[ordre]
        temps, debut = np.unique(-Ts, return_index=True)
        fins = np.r_[debut[1:], len(Ts)]
        S0 = 0.0
        S1 = np.zeros(p)
        S2 = np.zeros((p, p))
        for a, b in zip(debut, fins):
            xb, rb, eb = Xs[a:b], r[a:b], Es[a:b]
            S0 += rb.sum()
            S1 += rb @ xb
            S2 += xb.T @ (xb * rb[:, None])
            dmask = eb == 1
            nd = int(dmask.sum())
            if nd == 0:
                continue
            xd, rd = xb[dmask], rb[dmask]
            ll += eta[a:b][dmask].sum()
            grad += xd.sum(axis=0)
            if methode == "breslow" or nd == 1:
                ll -= nd * np.log(S0)
                m1 = S1 / S0
                grad -= nd * m1
                hess -= nd * (S2 / S0 - np.outer(m1, m1))
            else:
                S0d, S1d = rd.sum(), rd @ xd
                S2d = xd.T @ (xd * rd[:, None])
                f = np.arange(nd) / nd
                s0 = S0 - f * S0d
                s1 = S1[None, :] - f[:, None] * S1d[None, :]
                ll -= np.sum(np.log(s0))
                m1 = s1 / s0[:, None]
                grad -= m1.sum(axis=0)
                hess -= S2 * np.sum(1 / s0) - S2d * np.sum(f / s0) - m1.T @ m1
    return ll, grad, hess


@dataclass
class ResultatCox:
    """Résultat de :func:`modele_cox`.

    ``tableau`` contient ``coef``, ``erreur_std``, ``z``, ``p_valeur``,
    ``hr`` (rapport de risques instantanés = exp(coef)) et son IC.
    Un HR > 1 signifie que l'événement survient plus vite (risque instantané plus élevé).
    """

    tableau: pd.DataFrame
    indicateurs: dict
    encodeur: Encodeur
    explicatives: list[str]
    cov: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)
    T: np.ndarray = field(repr=False)
    E: np.ndarray = field(repr=False)
    strates: list = field(repr=False)
    moyennes: np.ndarray = field(repr=False)
    noms_strates: list = field(default_factory=list, repr=False)
    methode: str = "efron"
    _df_utilise: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def coefficients(self) -> pd.Series:
        """Coefficients (log des rapports de risques)."""
        return self.tableau["coef"]

    @property
    def hr(self) -> pd.Series:
        """Rapports de risques (hazard ratios)."""
        return self.tableau["hr"]

    def risque_cumule_base(self) -> dict:
        """Risque cumulé de base (estimateur de Breslow), par strate, pour des covariables centrées."""
        beta = self.tableau["coef"].to_numpy()
        out = {}
        Xc = self.X - self.moyennes
        for nom, idx in zip(self.noms_strates, self.strates):
            T, E, r = self.T[idx], self.E[idx], np.exp(Xc[idx] @ beta)
            temps = np.unique(T[E == 1])
            ordre = np.argsort(T)
            Ts, rs = T[ordre], r[ordre]
            cum_r = np.cumsum(rs[::-1])[::-1]
            pos = np.searchsorted(Ts, temps, side="left")
            denom = cum_r[pos]
            dk = np.bincount(np.searchsorted(temps, T[E == 1]), minlength=len(temps))
            out[nom] = pd.DataFrame({"temps": temps, "risque_cumule": np.cumsum(dk / denom)})
        return out

    def predire_survie(self, df: pd.DataFrame, temps: Sequence[float], strate: str | None = None) -> pd.DataFrame:
        """Probabilités de survie prédites ``S(t | x)`` pour chaque ligne de ``df``.

        Returns
        -------
        DataFrame (lignes = individus de ``df``, colonnes = ``temps``).
        """
        beta = self.tableau["coef"].to_numpy()
        X = self.encodeur.transform(df).to_numpy(dtype=float) - self.moyennes
        r = np.exp(X @ beta)
        bases = self.risque_cumule_base()
        temps = np.atleast_1d(np.asarray(temps, dtype=float))
        if len(bases) > 1 and strate is None:
            raise ValueError("Modèle stratifié : précisez `strate` (nom de colonne de df).")
        res = np.zeros((len(df), len(temps)))
        cles = df[strate].tolist() if strate else [next(iter(bases))] * len(df)
        for i, cle in enumerate(cles):
            base = bases[cle]
            j = np.searchsorted(base["temps"].to_numpy(), temps, side="right") - 1
            H0 = np.where(j >= 0, base["risque_cumule"].to_numpy()[np.maximum(j, 0)], 0.0)
            res[i] = np.exp(-H0 * r[i])
        return pd.DataFrame(res, index=df.index, columns=temps)

    def resume(self) -> str:
        """Résumé texte du modèle."""
        ind = self.indicateurs
        lignes = ["Modèle de Cox à risques proportionnels", "=" * 38,
                  f"Observations : {ind['n']} | événements : {ind['n_evenements']} | "
                  f"exclues (NA) : {ind['n_exclus']}",
                  f"Gestion des ex-aequo : {self.methode}",
                  f"Concordance (C de Harrell) : {ind['concordance']:.4f}",
                  f"Test du rapport de vraisemblance : chi2 = {ind['chi2_rv']:.2f}, "
                  f"ddl = {ind['ddl']}, p = {formater_p(ind['p_rv'])}",
                  f"AIC partiel : {ind['aic']:.2f}", ""]
        aff = self.tableau[["coef", "erreur_std", "p_valeur", "hr", "hr_ic_bas", "hr_ic_haut", "signif"]].copy()
        aff["p_valeur"] = aff["p_valeur"].map(formater_p)
        with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
            lignes.append(aff.to_string())
        if self.encodeur.references_:
            lignes += ["", "Références : " + ", ".join(f"{k} = {v}" for k, v in self.encodeur.references_.items())]
        return "\n".join(lignes)

    def __str__(self) -> str:  # pragma: no cover
        return self.resume()


def modele_cox(
    df: pd.DataFrame,
    duree: str,
    evenement: str,
    explicatives: Sequence[str] | None = None,
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    strate: str | None = None,
    methode: str = "efron",
    alpha: float = 0.05,
    max_iter: int = 50,
    reference_defaut: str = "frequente",
) -> ResultatCox:
    """Modèle de Cox à risques proportionnels.

    Parameters
    ----------
    df : DataFrame
    duree, evenement : str
        Colonnes durée et indicateur 0/1.
    explicatives : list of str, optional
        Covariables (``None`` = toutes les autres colonnes). Les variables
        catégorielles sont codées en indicatrices (référence paramétrable).
    reference : dict, optional
        Modalités de référence.
    interactions : list of tuple, optional
    strate : str, optional
        Variable de stratification : un risque de base distinct par strate
        (utile quand une variable ne respecte pas la proportionnalité, ex. le
        département ou la Mission Locale).
    methode : {"efron", "breslow"}
        Traitement des durées ex aequo (Efron est plus précis, défaut de R et lifelines).

    Returns
    -------
    ResultatCox

    Examples
    --------
    >>> cox = modele_cox(df, "duree", "sortie_emploi",
    ...                  ["sexe", "qpv", "niveau_formation"])        # doctest: +SKIP
    >>> print(cox.resume())                                         # doctest: +SKIP
    """
    if methode not in ("efron", "breslow"):
        raise ValueError("methode doit valoir 'efron' ou 'breslow'.")
    explicatives = explicatives_par_defaut(df, [duree, evenement], explicatives, exclure=[strate])
    d, T, E, n_exclus = _extraire(df, duree, evenement, [*explicatives, strate])
    if n_exclus:
        warnings.warn(f"{n_exclus} ligne(s) exclue(s) pour valeurs manquantes.", stacklevel=2)
    enc = Encodeur(explicatives, reference=reference, interactions=interactions, constante=False,
                   reference_defaut=reference_defaut)
    Xdf = enc.fit_transform(d)
    X = Xdf.to_numpy(dtype=float)
    moy = X.mean(axis=0)
    Xc = X - moy
    if strate:
        sv = d[strate].to_numpy()
        noms = sorted(pd.unique(sv).tolist(), key=lambda v: (str(type(v)), v))
        strates = [np.flatnonzero(sv == s) for s in noms]
    else:
        noms, strates = ["Ensemble"], [np.arange(len(T))]
    p = X.shape[1]
    beta = np.zeros(p)
    ll0, grad, hess = _cox_vraisemblance(beta, Xc, T, E, strates, methode)
    ll = ll0
    converge = False
    for _ in range(max_iter):
        try:
            pas = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            pas = np.linalg.pinv(-hess) @ grad
        t = 1.0
        while True:
            b_new = beta + t * pas
            ll_new, g_new, h_new = _cox_vraisemblance(b_new, Xc, T, E, strates, methode)
            if ll_new >= ll - 1e-9 or t < 1e-6:
                break
            t /= 2
        beta, ll_old = b_new, ll
        ll, grad, hess = ll_new, g_new, h_new
        if abs(ll - ll_old) < 1e-10 * (abs(ll) + 1) and np.max(np.abs(t * pas)) < 1e-7:
            converge = True
            break
    if not converge:
        warnings.warn("Le modèle de Cox n'a pas convergé (séparation possible).", stacklevel=2)
    cov = np.linalg.pinv(-hess)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    q = stats.norm.ppf(1 - alpha / 2)
    tab = pd.DataFrame({"coef": beta, "erreur_std": se, "z": z, "p_valeur": pv,
                        "ic_bas": beta - q * se, "ic_haut": beta + q * se},
                       index=pd.Index(Xdf.columns, name="terme"))
    tab["hr"] = np.exp(tab["coef"])
    tab["hr_ic_bas"] = np.exp(tab["ic_bas"])
    tab["hr_ic_haut"] = np.exp(tab["ic_haut"])
    tab["signif"] = tab["p_valeur"].map(etoiles)
    chi2 = 2 * (ll - ll0)
    risque = Xc @ beta
    ind = {"n": len(T), "n_evenements": int(E.sum()), "n_exclus": n_exclus, "loglik": ll,
           "loglik_nul": ll0, "chi2_rv": chi2, "ddl": p, "p_rv": float(stats.chi2.sf(chi2, p)),
           "aic": -2 * ll + 2 * p, "concordance": concordance(T, risque, E, strates=strates),
           "converge": converge,
           "wald_global": float(beta @ np.linalg.pinv(cov) @ beta)}
    return ResultatCox(tableau=tab, indicateurs=ind, encodeur=enc, explicatives=explicatives, cov=cov,
                       X=X, T=T, E=E, strates=strates, moyennes=moy, noms_strates=noms, methode=methode,
                       _df_utilise=d)


def concordance(T, risque, E, strates=None) -> float:
    """Indice de concordance de Harrell (C).

    Proportion de paires comparables pour lesquelles l'individu au risque prédit
    le plus élevé connaît l'événement le premier. 0,5 = aléatoire, 1 = parfait.
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    risque = np.asarray(risque, dtype=float)
    if strates is None:
        strates = [np.arange(len(T))]
    conc = disc = ties = 0.0
    for idx in strates:
        t, e, r = T[idx], E[idx], risque[idx]
        ordre = np.argsort(-t, kind="mergesort")
        t, e, r = t[ordre], e[ordre], r[ordre]
        vus: list = []  # risques des individus de durée strictement supérieure (triés)
        _, debut = np.unique(-t, return_index=True)
        fins = np.r_[debut[1:], len(t)]
        for a, b in zip(debut, fins):
            ev = np.flatnonzero(e[a:b] == 1) + a
            cens = np.flatnonzero(e[a:b] == 0) + a
            # les censurés à la même date sont comparables aux événements de cette date
            cens_r = sorted(r[cens])
            for i in ev:
                ri = r[i]
                for liste in (vus, cens_r):
                    n = len(liste)
                    if n == 0:
                        continue
                    lo = bisect.bisect_left(liste, ri)
                    hi = bisect.bisect_right(liste, ri)
                    conc += lo
                    ties += hi - lo
                    disc += n - hi
            for i in range(a, b):
                bisect.insort(vus, r[i])
    total = conc + disc + ties
    return float((conc + 0.5 * ties) / total) if total else np.nan


def test_proportionnalite(res: ResultatCox, transformation: str = "rang") -> pd.DataFrame:
    """Test de l'hypothèse des risques proportionnels (Grambsch & Therneau).

    Corrélation entre les résidus de Schoenfeld (mis à l'échelle) et une
    fonction du temps. Une p-valeur faible (< 0,05) indique que l'effet de la
    covariable varie dans le temps : envisager une stratification (``strate=``)
    ou une interaction avec le temps.

    Parameters
    ----------
    transformation : {"rang", "identite", "log", "km"}
        Fonction du temps utilisée (``"rang"`` par défaut, robuste).
    """
    beta = res.tableau["coef"].to_numpy()
    Xc = res.X - res.moyennes
    p = len(beta)
    residus, temps_ev = [], []
    for idx in res.strates:
        X, T, E = Xc[idx], res.T[idx], res.E[idx]
        r = np.exp(X @ beta)
        ordre = np.argsort(-T, kind="mergesort")
        X, T, E, r = X[ordre], T[ordre], E[ordre], r[ordre]
        S0 = np.cumsum(r)
        S1 = np.cumsum(X * r[:, None], axis=0)
        # pour chaque individu : indice du dernier ex aequo (inclusion de tout le groupe à risque)
        dernier = np.searchsorted(-T, -T, side="right") - 1
        for i in np.flatnonzero(E == 1):
            j = dernier[i]
            residus.append(X[i] - S1[j] / S0[j])
            temps_ev.append(T[i])
    R = np.asarray(residus)
    tv = np.asarray(temps_ev)
    if transformation == "rang":
        g = stats.rankdata(tv)
    elif transformation == "log":
        g = np.log(tv)
    elif transformation == "identite":
        g = tv
    elif transformation == "km":
        km = _km_un_groupe(res.T, res.E, np.ones(len(res.T)), 0.05)
        g = 1 - np.interp(tv, km["temps"], km["survie"])
    else:
        raise ValueError("transformation : 'rang', 'identite', 'log' ou 'km'.")
    gc = g - g.mean()
    nd = len(tv)
    V = res.cov
    U = gc @ R
    VU = V @ U
    denom = np.sum(gc ** 2)
    stat = nd * VU ** 2 / (np.diag(V) * denom)
    glob = float(nd * U @ V @ U / denom)
    out = pd.DataFrame({"terme": res.tableau.index, "chi2": stat, "ddl": 1,
                        "p_valeur": stats.chi2.sf(stat, 1)})
    out.loc[len(out)] = {"terme": "GLOBAL", "chi2": glob, "ddl": p, "p_valeur": stats.chi2.sf(glob, p)}
    out["signif"] = out["p_valeur"].map(etoiles)
    return out
