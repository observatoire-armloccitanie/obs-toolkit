"""Modèles de régression : linéaire, logistique (binaire et multinomiale), Poisson.

Toutes les fonctions suivent la même convention :

* ``df`` : le DataFrame d'analyse ;
* ``cible`` : la variable à expliquer ;
* ``explicatives`` : la liste des variables explicatives (``None`` = toutes les
  autres colonnes) ; les variables texte/catégorielles/booléennes sont
  automatiquement transformées en indicatrices, avec une modalité de référence
  paramétrable via ``reference={"variable": "modalité"}`` ;
* les lignes comportant une valeur manquante sur la cible ou une explicative
  sont exclues (le nombre d'exclusions est indiqué dans le résultat).

Examples
--------
>>> from obskit import regression, donnees_exemple
>>> df = donnees_exemple.generer_jeunes(2000)
>>> res = regression.regression_logistique(
...     df, cible="acces_emploi",
...     explicatives=["sexe", "age", "qpv", "niveau_formation"],
...     reference={"niveau_formation": "Niveau 3 (CAP-BEP)"})
>>> res.tableau[["odds_ratio", "p_valeur"]]      # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats

from ._encodage import Encodeur
from ._glm import AjustementGLM, ajuster_glm, tableau_coefficients
from ._utils import en_liste, etoiles, explicatives_par_defaut, formater_p, verifier_colonnes

__all__ = [
    "ResultatRegression",
    "ResultatMultinomial",
    "regression_lineaire",
    "regression_logistique",
    "regression_poisson",
    "regression_logistique_multinomiale",
    "effets_marginaux_moyens",
    "test_global_variables",
    "test_rapport_vraisemblance",
    "comparer_modeles",
    "performance_classification",
    "test_hosmer_lemeshow",
    "vif",
    "selection_pas_a_pas",
    "auc",
]

_FAMILLE = {"lineaire": "gaussienne", "logistique": "binomiale", "poisson": "poisson"}


# ----------------------------------------------------------------------------
# Résultat
# ----------------------------------------------------------------------------
@dataclass
class ResultatRegression:
    """Résultat d'une régression (linéaire, logistique ou de Poisson).

    Attributes
    ----------
    type_modele : str
        ``"lineaire"``, ``"logistique"`` ou ``"poisson"``.
    cible : str
        Variable expliquée.
    tableau : pandas.DataFrame
        Un terme par ligne : ``coef``, ``erreur_std``, ``stat``, ``p_valeur``,
        ``ic_bas``, ``ic_haut``, ``signif`` et, pour les modèles logistique et
        de Poisson, ``odds_ratio`` / ``irr`` avec leur intervalle de confiance.
    indicateurs : dict
        Qualité d'ajustement (n, R², AIC, BIC, log-vraisemblance, pseudo-R²…).
    encodeur : Encodeur
        Encodage des variables (références, modalités) réutilisé pour prédire.
    """

    type_modele: str
    cible: str
    explicatives: list[str]
    tableau: pd.DataFrame
    indicateurs: dict
    encodeur: Encodeur
    ajustement: AjustementGLM = field(repr=False)
    y: np.ndarray = field(repr=False)
    X: pd.DataFrame = field(repr=False)
    poids: np.ndarray | None = field(default=None, repr=False)
    offset: np.ndarray | None = field(default=None, repr=False)
    modalite_positive: object = None
    alpha: float = 0.05
    _df_utilise: pd.DataFrame | None = field(default=None, repr=False)

    # -- accès pratiques ---------------------------------------------------
    @property
    def coefficients(self) -> pd.Series:
        """Coefficients estimés (échelle du prédicteur linéaire)."""
        return self.tableau["coef"]

    @property
    def p_valeurs(self) -> pd.Series:
        """p-valeurs des coefficients."""
        return self.tableau["p_valeur"]

    @property
    def valeurs_ajustees(self) -> pd.Series:
        """Valeurs prédites sur l'échantillon d'estimation (échelle de la réponse)."""
        return pd.Series(self.ajustement.mu, index=self.X.index, name="ajuste")

    @property
    def residus(self) -> pd.Series:
        """Résidus bruts ``y - ŷ``."""
        return pd.Series(self.y - self.ajustement.mu, index=self.X.index, name="residu")

    def residus_pearson(self) -> pd.Series:
        """Résidus de Pearson."""
        mu = self.ajustement.mu
        var = {"lineaire": np.ones_like(mu), "logistique": mu * (1 - mu), "poisson": mu}[self.type_modele]
        return pd.Series((self.y - mu) / np.sqrt(var), index=self.X.index, name="residu_pearson")

    # -- prédiction --------------------------------------------------------
    def predire(self, df: pd.DataFrame, echelle: str = "reponse") -> pd.Series:
        """Prédit sur de nouvelles données.

        Parameters
        ----------
        df : DataFrame
            Doit contenir les variables explicatives.
        echelle : {"reponse", "lien"}
            ``"reponse"`` : moyenne prédite (probabilité pour la logistique,
            comptage attendu pour Poisson) ; ``"lien"`` : prédicteur linéaire.
        """
        X = self.encodeur.transform(df).to_numpy(dtype=float)
        eta = X @ self.tableau["coef"].to_numpy()
        if echelle == "lien" or self.type_modele == "lineaire":
            return pd.Series(eta, index=df.index, name="prediction")
        if self.type_modele == "logistique":
            return pd.Series(special.expit(eta), index=df.index, name="probabilite")
        return pd.Series(np.exp(eta), index=df.index, name="prediction")

    # -- tableaux ----------------------------------------------------------
    def tableau_complet(self) -> pd.DataFrame:
        """Tableau de coefficients incluant les lignes « référence » des variables catégorielles.

        Format usuel des publications : chaque variable est suivie de toutes ses
        modalités, la modalité de référence portant la mention ``"Réf."``.
        """
        lignes = []
        tab = self.tableau
        if "const" in tab.index:
            lignes.append(("(Constante)", "", *tab.loc["const"].tolist()))
        for var in self.explicatives:
            if var in self.encodeur.modalites_:
                ref = self.encodeur.references_[var]
                for m in self.encodeur.modalites_[var]:
                    if m == ref:
                        ligne = [np.nan] * tab.shape[1]
                        if "signif" in tab.columns:
                            ligne[list(tab.columns).index("signif")] = "Réf."
                        for c in ("odds_ratio", "irr"):
                            if c in tab.columns:
                                ligne[list(tab.columns).index(c)] = 1.0
                        lignes.append((var, m, *ligne))
                    else:
                        lignes.append((var, m, *tab.loc[f"{var}[{m}]"].tolist()))
            else:
                lignes.append((var, "", *tab.loc[var].tolist()))
        for a, b in self.encodeur.interactions:
            for col in self.encodeur.colonnes_par_variable_[f"{a}:{b}"]:
                lignes.append((f"{a} x {b}", col, *tab.loc[col].tolist()))
        out = pd.DataFrame(lignes, columns=["variable", "modalite", *tab.columns])
        return out

    def resume(self) -> str:
        """Résumé texte du modèle (à afficher avec ``print``)."""
        ind = self.indicateurs
        titre = {"lineaire": "Régression linéaire (MCO)", "logistique": "Régression logistique",
                 "poisson": "Régression de Poisson"}[self.type_modele]
        lignes = [titre, "=" * len(titre), f"Variable expliquée : {self.cible}"]
        if self.modalite_positive is not None:
            lignes.append(f"Modalité modélisée : {self.modalite_positive}")
        lignes.append(f"Observations : {ind['n']} (exclues pour valeurs manquantes : {ind['n_exclus']})")
        lignes.append(f"Erreurs-types : {self.ajustement.type_cov}")
        for cle, lib in [("r2", "R²"), ("r2_ajuste", "R² ajusté"), ("pseudo_r2_mcfadden", "Pseudo-R² (McFadden)"),
                         ("auc", "AUC"), ("aic", "AIC"), ("bic", "BIC"), ("loglik", "Log-vraisemblance")]:
            if cle in ind and ind[cle] is not None:
                lignes.append(f"{lib:<22}: {ind[cle]:.4f}")
        if "p_test_global" in ind:
            lignes.append(f"Test global (Wald)    : p = {formater_p(ind['p_test_global'])}")
        lignes.append("")
        cols = ["coef", "erreur_std", "p_valeur"]
        for c in ("odds_ratio", "irr"):
            if c in self.tableau.columns:
                cols += [c, f"{c}_ic_bas", f"{c}_ic_haut"]
        aff = self.tableau[cols].copy()
        aff["p_valeur"] = aff["p_valeur"].map(formater_p)
        aff["signif"] = self.tableau["signif"]
        with pd.option_context("display.width", 140, "display.max_columns", 20,
                               "display.float_format", "{:.4f}".format):
            lignes.append(aff.to_string())
        refs = self.encodeur.references_
        if refs:
            lignes.append("")
            lignes.append("Références : " + ", ".join(f"{k} = {v}" for k, v in refs.items()))
        lignes.append("Signif. : *** <0.001 ; ** <0.01 ; * <0.05 ; . <0.1")
        return "\n".join(lignes)

    def __str__(self) -> str:  # pragma: no cover - affichage
        return self.resume()


# ----------------------------------------------------------------------------
# Préparation commune
# ----------------------------------------------------------------------------
def _preparer(df, cible, explicatives, autres):
    verifier_colonnes(df, [cible, *explicatives, *[a for a in autres if a]])
    cols = list(dict.fromkeys([cible, *explicatives, *[a for a in autres if a]]))
    sous = df[cols]
    masque = sous.notna().all(axis=1)
    n_exclus = int((~masque).sum())
    if n_exclus:
        warnings.warn(f"{n_exclus} ligne(s) exclue(s) pour valeurs manquantes.", stacklevel=3)
    return df.loc[masque], n_exclus


def _cible_binaire(serie: pd.Series, modalite_positive):
    valeurs = pd.unique(serie.dropna())
    if modalite_positive is not None:
        if modalite_positive not in set(valeurs):
            raise ValueError(f"Modalité positive '{modalite_positive}' absente. Valeurs : {list(valeurs)}")
        return (serie == modalite_positive).astype(float).to_numpy(), modalite_positive
    if len(valeurs) != 2:
        raise ValueError(
            f"La cible doit avoir exactement 2 modalités (trouvé {len(valeurs)} : {list(valeurs)[:10]}). "
            "Utilisez `modalite_positive=` ou la régression multinomiale."
        )
    ens = set(valeurs.tolist())
    if ens <= {0, 1} or ens <= {True, False}:
        return serie.astype(float).to_numpy(), 1
    raise ValueError(
        f"Cible non codée 0/1 ({list(valeurs)}) : précisez `modalite_positive=` (ex. '{valeurs[0]}')."
    )


def _ajuster(
    type_modele, df, cible, explicatives, reference, interactions, poids, erreurs, grappes,
    offset_col=None, modalite_positive=None, alpha=0.05, reference_defaut="frequente",
    categorielles=None,
) -> ResultatRegression:
    explicatives = explicatives_par_defaut(df, cible, explicatives, exclure=[poids, grappes, offset_col])
    if not explicatives:
        raise ValueError("Aucune variable explicative.")
    d, n_exclus = _preparer(df, cible, explicatives, [poids, grappes, offset_col])
    if len(d) == 0:
        raise ValueError("Aucune observation complète après exclusion des valeurs manquantes.")
    enc = Encodeur(explicatives, reference=reference, interactions=interactions,
                   reference_defaut=reference_defaut, categorielles=categorielles)
    X = enc.fit_transform(d)
    famille = _FAMILLE[type_modele]
    if type_modele == "logistique":
        y, modalite_positive = _cible_binaire(d[cible], modalite_positive)
    else:
        y = pd.to_numeric(d[cible], errors="raise").to_numpy(dtype=float)
    w = d[poids].to_numpy(dtype=float) if poids else None
    off = None
    if offset_col:
        expo = d[offset_col].to_numpy(dtype=float)
        if np.any(expo <= 0):
            raise ValueError("L'exposition doit être strictement positive.")
        off = np.log(expo)
    g = d[grappes].to_numpy() if grappes else None
    type_cov = "cluster" if grappes and erreurs in (None, "classique", "cluster") else (erreurs or "classique")
    aj = ajuster_glm(X.to_numpy(), y, famille=famille, poids=w, offset=off, type_cov=type_cov, grappes=g)
    if not aj.converge:
        warnings.warn("L'algorithme n'a pas convergé.", stacklevel=2)
    tab = tableau_coefficients(aj, list(X.columns), alpha=alpha)
    if type_modele in ("logistique", "poisson"):
        nom = "odds_ratio" if type_modele == "logistique" else "irr"
        tab[nom] = np.exp(tab["coef"])
        tab[f"{nom}_ic_bas"] = np.exp(tab["ic_bas"])
        tab[f"{nom}_ic_haut"] = np.exp(tab["ic_haut"])
    tab["signif"] = tab["p_valeur"].map(etoiles)

    # indicateurs
    ind: dict = {"n": aj.n, "n_exclus": n_exclus, "k": aj.p, "loglik": aj.loglik,
                 "aic": -2 * aj.loglik + 2 * aj.p, "bic": -2 * aj.loglik + np.log(aj.n) * aj.p,
                 "deviance": aj.deviance, "converge": aj.converge}
    wv = np.ones(aj.n) if w is None else w
    if "const" in X.columns:
        nul = ajuster_glm(np.ones((aj.n, 1)), y, famille=famille, poids=w, offset=off)
        ind["loglik_nul"] = nul.loglik
        if type_modele == "lineaire":
            ybar = np.average(y, weights=wv)
            tss = np.sum(wv * (y - ybar) ** 2)
            rss = np.sum(wv * (y - aj.mu) ** 2)
            ind["r2"] = 1 - rss / tss
            ind["r2_ajuste"] = 1 - (1 - ind["r2"]) * (aj.n - 1) / (aj.n - aj.p)
            ind["sigma"] = float(np.sqrt(aj.dispersion))
        else:
            ind["pseudo_r2_mcfadden"] = 1 - aj.loglik / nul.loglik
            ind["chi2_rv"] = 2 * (aj.loglik - nul.loglik)
            ind["p_rv"] = float(stats.chi2.sf(ind["chi2_rv"], aj.p - 1)) if aj.p > 1 else np.nan
        # test global de Wald (tous les coefficients hors constante)
        idx = [i for i, c in enumerate(X.columns) if c != "const"]
        if idx:
            b = aj.coef[idx]
            V = aj.cov[np.ix_(idx, idx)]
            wald = float(b @ np.linalg.pinv(V) @ b)
            q = len(idx)
            if type_modele == "lineaire":
                ind["f_test_global"] = wald / q
                ind["p_test_global"] = float(stats.f.sf(wald / q, q, aj.ddl_resid))
            else:
                ind["wald_test_global"] = wald
                ind["p_test_global"] = float(stats.chi2.sf(wald, q))
    if type_modele == "logistique":
        ind["auc"] = auc(y, aj.mu, poids=w)
    if type_modele == "poisson":
        ind["surdispersion"] = float(np.sum(wv * (y - aj.mu) ** 2 / aj.mu) / aj.ddl_resid)
        if ind["surdispersion"] > 1.5:
            warnings.warn(
                f"Surdispersion détectée (ratio = {ind['surdispersion']:.2f}) : utilisez "
                "erreurs='HC1' (quasi-Poisson robuste).", stacklevel=2)
    return ResultatRegression(
        type_modele=type_modele, cible=cible, explicatives=explicatives, tableau=tab,
        indicateurs=ind, encodeur=enc, ajustement=aj, y=y, X=X, poids=w, offset=off,
        modalite_positive=modalite_positive if type_modele == "logistique" else None,
        alpha=alpha, _df_utilise=d,
    )


# ----------------------------------------------------------------------------
# Fonctions publiques
# ----------------------------------------------------------------------------
def regression_lineaire(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    poids: str | None = None,
    erreurs: str = "classique",
    grappes: str | None = None,
    alpha: float = 0.05,
    reference_defaut: str = "frequente",
    categorielles: Sequence[str] | None = None,
) -> ResultatRegression:
    """Régression linéaire par moindres carrés (pondérés le cas échéant).

    Parameters
    ----------
    df : DataFrame
        Données.
    cible : str
        Variable quantitative à expliquer (ex. ``"duree_accompagnement"``).
    explicatives : list of str, optional
        Variables explicatives ; ``None`` = toutes les autres colonnes.
    reference : dict, optional
        Modalités de référence, ex. ``{"sexe": "Femme"}``.
    interactions : list of tuple, optional
        Interactions, ex. ``[("sexe", "qpv")]``.
    poids : str, optional
        Colonne de pondération.
    erreurs : {"classique", "HC0", "HC1", "HC3", "cluster"}
        Type d'erreurs-types. ``"HC3"`` est recommandé en cas d'hétéroscédasticité.
    grappes : str, optional
        Colonne identifiant les grappes (ex. ``"mission_locale"``) : active des
        erreurs-types robustes par grappe.
    alpha : float
        Risque pour les intervalles de confiance (0,05 → IC à 95 %).
    reference_defaut : {"frequente", "premiere"}
        Règle de choix des références non précisées.
    categorielles : list of str, optional
        Variables numériques à traiter comme catégorielles (ex. ``"annee"``).

    Returns
    -------
    ResultatRegression
    """
    return _ajuster("lineaire", df, cible, explicatives, reference, interactions, poids,
                    erreurs, grappes, alpha=alpha, reference_defaut=reference_defaut,
                    categorielles=categorielles)


def regression_logistique(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    modalite_positive: object = None,
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    poids: str | None = None,
    erreurs: str = "classique",
    grappes: str | None = None,
    alpha: float = 0.05,
    reference_defaut: str = "frequente",
    categorielles: Sequence[str] | None = None,
) -> ResultatRegression:
    """Régression logistique binaire (odds ratios, IC, AUC, pseudo-R²).

    Parameters
    ----------
    cible : str
        Variable binaire. Si elle n'est pas codée 0/1 ou booléenne, préciser
        ``modalite_positive`` (ex. ``"Emploi"``).
    modalite_positive : optional
        Modalité de la cible modélisée (probabilité de ...).

    Autres paramètres : voir :func:`regression_lineaire`.

    Notes
    -----
    Les odds ratios ne sont pas comparables d'un modèle à l'autre (ils dépendent
    de l'hétérogénéité non observée) : pour comparer des effets entre modèles ou
    groupes, préférer les effets marginaux moyens (:func:`effets_marginaux_moyens`).
    """
    return _ajuster("logistique", df, cible, explicatives, reference, interactions, poids,
                    erreurs, grappes, modalite_positive=modalite_positive, alpha=alpha,
                    reference_defaut=reference_defaut, categorielles=categorielles)


def regression_poisson(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    exposition: str | None = None,
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    poids: str | None = None,
    erreurs: str = "classique",
    grappes: str | None = None,
    alpha: float = 0.05,
    reference_defaut: str = "frequente",
    categorielles: Sequence[str] | None = None,
) -> ResultatRegression:
    """Régression de Poisson pour des comptages (nombre d'entretiens, de PMSMP…).

    Parameters
    ----------
    exposition : str, optional
        Colonne de durée d'exposition (ex. mois d'accompagnement) : le modèle
        estime alors des taux (``log(exposition)`` en *offset*).

    Les coefficients exponentiés (``irr``) sont des rapports de taux. En cas de
    surdispersion, utiliser ``erreurs="HC1"``.
    """
    return _ajuster("poisson", df, cible, explicatives, reference, interactions, poids,
                    erreurs, grappes, offset_col=exposition, alpha=alpha,
                    reference_defaut=reference_defaut, categorielles=categorielles)


# ----------------------------------------------------------------------------
# Logistique multinomiale
# ----------------------------------------------------------------------------
@dataclass
class ResultatMultinomial:
    """Résultat d'une régression logistique multinomiale.

    ``tableau`` est indexé par (modalité de la cible, terme) et contient les
    coefficients, erreurs-types, p-valeurs et rapports de risques relatifs
    (``rrr`` = exp(coef)) par rapport à la modalité de référence de la cible.
    """

    cible: str
    modalites: list
    reference_cible: object
    tableau: pd.DataFrame
    indicateurs: dict
    encodeur: Encodeur
    coef: np.ndarray = field(repr=False)

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probabilités prédites de chaque modalité de la cible."""
        X = self.encodeur.transform(df).to_numpy(dtype=float)
        eta = np.column_stack([np.zeros(len(X)), X @ self.coef])
        P = special.softmax(eta, axis=1)
        autres = [m for m in self.modalites if m != self.reference_cible]
        return pd.DataFrame(P, index=df.index, columns=[self.reference_cible, *autres])[self.modalites]

    def resume(self) -> str:
        """Résumé texte."""
        lignes = ["Régression logistique multinomiale", "=" * 34,
                  f"Cible : {self.cible} (référence : {self.reference_cible})",
                  f"Observations : {self.indicateurs['n']}",
                  f"Pseudo-R² (McFadden) : {self.indicateurs['pseudo_r2_mcfadden']:.4f}",
                  f"AIC : {self.indicateurs['aic']:.2f}", ""]
        with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
            lignes.append(self.tableau[["coef", "erreur_std", "p_valeur", "rrr", "signif"]].to_string())
        return "\n".join(lignes)


def regression_logistique_multinomiale(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    reference_cible: object = None,
    reference: Mapping[str, object] | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    poids: str | None = None,
    alpha: float = 0.05,
    max_iter: int = 100,
    reference_defaut: str = "frequente",
) -> ResultatMultinomial:
    """Régression logistique multinomiale (cible à plus de deux modalités non ordonnées).

    Typiquement : situation à la sortie (emploi / formation / sans solution…).

    Parameters
    ----------
    reference_cible : optional
        Modalité de référence de la cible (défaut : la plus fréquente).

    Autres paramètres : voir :func:`regression_lineaire`.
    """
    explicatives = explicatives_par_defaut(df, cible, explicatives, exclure=[poids])
    d, n_exclus = _preparer(df, cible, explicatives, [poids])
    enc = Encodeur(explicatives, reference=reference, interactions=interactions,
                   reference_defaut=reference_defaut)
    Xdf = enc.fit_transform(d)
    X = Xdf.to_numpy(dtype=float)
    n, p = X.shape
    w = d[poids].to_numpy(dtype=float) if poids else np.ones(n)
    y = d[cible]
    modalites = sorted(y.unique().tolist(), key=lambda v: (str(type(v)), v))
    if len(modalites) < 3:
        warnings.warn("Moins de 3 modalités : une régression logistique binaire suffit.", stacklevel=2)
    if reference_cible is None:
        reference_cible = y.value_counts().index[0]
    if reference_cible not in modalites:
        raise ValueError(f"reference_cible '{reference_cible}' absente : {modalites}")
    autres = [m for m in modalites if m != reference_cible]
    K = len(autres)
    Y = np.column_stack([(y == m).to_numpy(dtype=float) for m in autres])
    B = np.zeros((p, K))

    def loglik(Bm):
        eta = np.column_stack([np.zeros(n), X @ Bm])
        return float(np.sum(w * (np.sum(Y * eta[:, 1:], axis=1) - special.logsumexp(eta, axis=1))))

    ll = loglik(B)
    converge = False
    for _ in range(max_iter):
        eta = np.column_stack([np.zeros(n), X @ B])
        P = special.softmax(eta, axis=1)[:, 1:]
        grad = (X.T @ (w[:, None] * (Y - P))).reshape(-1, order="F")
        H = np.zeros((p * K, p * K))
        for j in range(K):
            for k in range(j, K):
                wjk = w * P[:, j] * ((1.0 if j == k else 0.0) - P[:, k])
                bloc = X.T @ (X * wjk[:, None])
                H[j * p:(j + 1) * p, k * p:(k + 1) * p] = bloc
                H[k * p:(k + 1) * p, j * p:(j + 1) * p] = bloc
        pas = np.linalg.solve(H, grad).reshape((p, K), order="F")
        t = 1.0
        while True:
            B_new = B + t * pas
            ll_new = loglik(B_new)
            if ll_new >= ll - 1e-10 or t < 1e-8:
                break
            t /= 2
        B, ll_old, ll = B_new, ll, ll_new
        if abs(ll - ll_old) < 1e-10 * (abs(ll) + 1):
            converge = True
            break
    # information finale
    eta = np.column_stack([np.zeros(n), X @ B])
    P = special.softmax(eta, axis=1)[:, 1:]
    H = np.zeros((p * K, p * K))
    for j in range(K):
        for k in range(K):
            wjk = w * P[:, j] * ((1.0 if j == k else 0.0) - P[:, k])
            H[j * p:(j + 1) * p, k * p:(k + 1) * p] = X.T @ (X * wjk[:, None])
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape((p, K), order="F")
    z = B / se
    pv = 2 * stats.norm.sf(np.abs(z))
    q = stats.norm.ppf(1 - alpha / 2)
    lignes = []
    for k, m in enumerate(autres):
        for i, terme in enumerate(Xdf.columns):
            lignes.append((m, terme, B[i, k], se[i, k], z[i, k], pv[i, k],
                           B[i, k] - q * se[i, k], B[i, k] + q * se[i, k]))
    tab = pd.DataFrame(lignes, columns=["modalite", "terme", "coef", "erreur_std", "stat",
                                        "p_valeur", "ic_bas", "ic_haut"]).set_index(["modalite", "terme"])
    tab["rrr"] = np.exp(tab["coef"])
    tab["rrr_ic_bas"] = np.exp(tab["ic_bas"])
    tab["rrr_ic_haut"] = np.exp(tab["ic_haut"])
    tab["signif"] = tab["p_valeur"].map(etoiles)
    freq = np.array([np.sum(w * (y == m).to_numpy()) for m in modalites]) / np.sum(w)
    ll0 = float(np.sum(w * np.log(freq[[modalites.index(v) for v in y]])))
    k_par = p * K
    ind = {"n": n, "n_exclus": n_exclus, "loglik": ll, "loglik_nul": ll0,
           "pseudo_r2_mcfadden": 1 - ll / ll0, "aic": -2 * ll + 2 * k_par,
           "bic": -2 * ll + np.log(n) * k_par, "converge": converge}
    return ResultatMultinomial(cible=cible, modalites=modalites, reference_cible=reference_cible,
                               tableau=tab, indicateurs=ind, encodeur=enc, coef=B)


# ----------------------------------------------------------------------------
# Post-estimation
# ----------------------------------------------------------------------------
def effets_marginaux_moyens(
    res: ResultatRegression, variables: Sequence[str] | None = None, h: float = 1e-4,
) -> pd.DataFrame:
    """Effets marginaux moyens (AME) avec erreurs-types par la méthode delta.

    * variable numérique : dérivée moyenne de la prédiction (échelle de la
      réponse) par rapport à la variable ;
    * variable catégorielle : écart moyen de prédiction entre chaque modalité et
      la référence (toutes choses égales par ailleurs).

    Pour une logistique, l'AME s'interprète en points de probabilité (x100 =
    points de pourcentage). Les interactions sont prises en compte.

    Parameters
    ----------
    res : ResultatRegression
    variables : list of str, optional
        Variables à traiter (défaut : toutes les explicatives).
    """
    d = res._df_utilise
    beta = res.tableau["coef"].to_numpy()
    enc = res.encodeur
    w = np.ones(len(d)) if res.poids is None else res.poids
    off = np.zeros(len(d)) if res.offset is None else res.offset

    def moyenne_pred(dfm, b):
        eta = enc.transform(dfm).to_numpy(dtype=float) @ b + off
        if res.type_modele == "logistique":
            mu = special.expit(eta)
        elif res.type_modele == "poisson":
            mu = np.exp(eta)
        else:
            mu = eta
        return float(np.average(mu, weights=w))

    def ame_et_grad(f):
        val = f(beta)
        grad = np.zeros(len(beta))
        for j in range(len(beta)):
            e = np.zeros(len(beta))
            pas = 1e-6 * max(1.0, abs(beta[j]))
            e[j] = pas
            grad[j] = (f(beta + e) - f(beta - e)) / (2 * pas)
        return val, grad

    lignes = []
    for var in en_liste(variables) or res.explicatives:
        if var in enc.modalites_:
            ref = enc.references_[var]
            d_ref = d.copy()
            d_ref[var] = ref
            for m in enc.modalites_[var]:
                if m == ref:
                    continue
                d_m = d.copy()
                d_m[var] = m
                val, g = ame_et_grad(lambda b, dm=d_m: moyenne_pred(dm, b) - moyenne_pred(d_ref, b))
                lignes.append((var, f"{m} (vs {ref})", val, g))
        else:
            x = d[var].astype(float)
            pas = h * max(1.0, float(x.std()))
            d_p, d_m = d.copy(), d.copy()
            d_p[var] = x + pas
            d_m[var] = x - pas
            val, g = ame_et_grad(lambda b: (moyenne_pred(d_p, b) - moyenne_pred(d_m, b)) / (2 * pas))
            lignes.append((var, "(dérivée)", val, g))
    q = stats.norm.ppf(1 - res.alpha / 2)
    sortie = []
    for var, mod, val, g in lignes:
        se = float(np.sqrt(g @ res.ajustement.cov @ g))
        z = val / se if se > 0 else np.nan
        sortie.append({"variable": var, "modalite": mod, "ame": val, "erreur_std": se,
                       "p_valeur": float(2 * stats.norm.sf(abs(z))), "ic_bas": val - q * se,
                       "ic_haut": val + q * se})
    out = pd.DataFrame(sortie)
    out["signif"] = out["p_valeur"].map(etoiles)
    return out


def test_global_variables(res: ResultatRegression) -> pd.DataFrame:
    """Test de Wald de nullité jointe de tous les coefficients de chaque variable.

    Indique si une variable catégorielle (toutes modalités confondues) a un
    effet significatif, ce que les p-valeurs modalité par modalité ne disent pas.
    """
    cols = list(res.X.columns)
    lignes = []
    for var, termes in res.encodeur.colonnes_par_variable_.items():
        idx = [cols.index(t) for t in termes]
        b = res.ajustement.coef[idx]
        V = res.ajustement.cov[np.ix_(idx, idx)]
        wald = float(b @ np.linalg.pinv(V) @ b)
        lignes.append({"variable": var, "ddl": len(idx), "wald_chi2": wald,
                       "p_valeur": float(stats.chi2.sf(wald, len(idx)))})
    out = pd.DataFrame(lignes)
    out["signif"] = out["p_valeur"].map(etoiles)
    return out


def test_rapport_vraisemblance(restreint: ResultatRegression, complet: ResultatRegression) -> dict:
    """Test du rapport de vraisemblance entre deux modèles emboîtés.

    Les deux modèles doivent être estimés sur les mêmes observations.
    """
    if restreint.indicateurs["n"] != complet.indicateurs["n"]:
        raise ValueError("Les modèles n'ont pas le même nombre d'observations : "
                         "estimez-les sur le même échantillon (dropna préalable).")
    ddl = complet.indicateurs["k"] - restreint.indicateurs["k"]
    if ddl <= 0:
        raise ValueError("Le modèle `complet` doit avoir plus de paramètres que `restreint`.")
    stat = 2 * (complet.indicateurs["loglik"] - restreint.indicateurs["loglik"])
    return {"chi2": stat, "ddl": ddl, "p_valeur": float(stats.chi2.sf(stat, ddl))}


def comparer_modeles(modeles: Mapping[str, ResultatRegression]) -> pd.DataFrame:
    """Tableau comparatif (n, nb de paramètres, log-vraisemblance, AIC, BIC, R²/pseudo-R², AUC)."""
    lignes = {}
    for nom, r in modeles.items():
        ind = r.indicateurs
        lignes[nom] = {k: ind.get(k) for k in ("n", "k", "loglik", "aic", "bic", "r2",
                                                "r2_ajuste", "pseudo_r2_mcfadden", "auc")}
    return pd.DataFrame(lignes).T.dropna(axis=1, how="all")


def auc(y, score, poids=None) -> float:
    """Aire sous la courbe ROC (statistique de Mann-Whitney), pondérée si besoin."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    w = np.ones_like(y) if poids is None else np.asarray(poids, dtype=float)
    if poids is None:
        rangs = stats.rankdata(s)
        n1 = y.sum()
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        return float((rangs[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    ordre = np.argsort(s, kind="mergesort")
    s, y, w = s[ordre], y[ordre], w[ordre]
    _, debut, cnt = np.unique(s, return_index=True, return_counts=True)
    w1 = np.add.reduceat(w * y, debut)
    w0 = np.add.reduceat(w * (1 - y), debut)
    cum0 = np.cumsum(w0) - w0
    num = np.sum(w1 * (cum0 + 0.5 * w0))
    return float(num / (w1.sum() * w0.sum()))


def performance_classification(res: ResultatRegression, seuil: float = 0.5) -> dict:
    """Indicateurs de classement d'une logistique : matrice de confusion, sensibilité…"""
    if res.type_modele != "logistique":
        raise ValueError("Réservé aux régressions logistiques.")
    y = res.y.astype(int)
    p = res.ajustement.mu
    yhat = (p >= seuil).astype(int)
    vp = int(np.sum((y == 1) & (yhat == 1)))
    vn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    sens = vp / (vp + fn) if vp + fn else np.nan
    spec = vn / (vn + fp) if vn + fp else np.nan
    prec = vp / (vp + fp) if vp + fp else np.nan
    return {
        "matrice_confusion": pd.DataFrame([[vn, fp], [fn, vp]], index=["obs_0", "obs_1"],
                                          columns=["pred_0", "pred_1"]),
        "exactitude": (vp + vn) / len(y), "sensibilite": sens, "specificite": spec,
        "precision": prec, "f1": 2 * prec * sens / (prec + sens) if prec + sens else np.nan,
        "auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)),
    }


def test_hosmer_lemeshow(res: ResultatRegression, groupes: int = 10) -> dict:
    """Test d'adéquation de Hosmer-Lemeshow (logistique). p > 0,05 : pas d'inadéquation détectée."""
    if res.type_modele != "logistique":
        raise ValueError("Réservé aux régressions logistiques.")
    d = pd.DataFrame({"y": res.y, "p": res.ajustement.mu})
    d["g"] = pd.qcut(d["p"].rank(method="first"), groupes, labels=False)
    agg = d.groupby("g").agg(obs=("y", "sum"), att=("p", "sum"), n=("y", "size"))
    stat = float(np.sum((agg.obs - agg.att) ** 2 / (agg.att * (1 - agg.att / agg.n))))
    ddl = groupes - 2
    return {"chi2": stat, "ddl": ddl, "p_valeur": float(stats.chi2.sf(stat, ddl)), "tableau": agg}


def vif(
    df: pd.DataFrame, explicatives: Sequence[str], reference: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Facteurs d'inflation de la variance (multicolinéarité).

    Renvoie, par variable, le VIF généralisé (GVIF de Fox & Monette) et
    ``GVIF^(1/(2·ddl))``, comparable à la racine d'un VIF classique. Règle
    usuelle : ``GVIF^(1/(2·ddl)) > 2,2`` (VIF > 5) signale une colinéarité forte.
    """
    explicatives = en_liste(explicatives)
    d = df[explicatives].dropna()
    enc = Encodeur(explicatives, reference=reference, constante=False)
    X = enc.fit_transform(d)
    R = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    R = np.atleast_2d(R)
    det_R = np.linalg.det(R)
    cols = list(X.columns)
    lignes = []
    for var in explicatives:
        idx = [cols.index(c) for c in enc.colonnes_par_variable_[var]]
        autres = [i for i in range(len(cols)) if i not in idx]
        det_a = np.linalg.det(R[np.ix_(idx, idx)])
        det_b = np.linalg.det(R[np.ix_(autres, autres)]) if autres else 1.0
        gvif = det_a * det_b / det_R
        lignes.append({"variable": var, "ddl": len(idx), "gvif": gvif,
                       "gvif_ajuste": gvif ** (1 / (2 * len(idx)))})
    return pd.DataFrame(lignes)


def selection_pas_a_pas(
    df: pd.DataFrame,
    cible: str,
    candidates: Sequence[str],
    type_modele: str = "logistique",
    critere: str = "aic",
    forcees: Sequence[str] | None = None,
    **kwargs,
) -> tuple[ResultatRegression, pd.DataFrame]:
    """Sélection ascendante de variables selon l'AIC ou le BIC.

    À utiliser comme outil exploratoire : la sélection automatique biaise les
    p-valeurs du modèle final. Les variables de ``forcees`` sont toujours incluses.

    Returns
    -------
    (ResultatRegression, DataFrame)
        Le modèle retenu et l'historique des étapes.
    """
    if critere not in ("aic", "bic"):
        raise ValueError("critere doit valoir 'aic' ou 'bic'.")
    fonctions = {"lineaire": regression_lineaire, "logistique": regression_logistique,
                 "poisson": regression_poisson}
    f = fonctions[type_modele]
    forcees = en_liste(forcees)
    candidates = [c for c in en_liste(candidates) if c not in forcees]
    d = df[[cible, *forcees, *candidates]].dropna()
    retenues = list(forcees)
    histo = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if retenues:
            meilleur = f(d, cible, retenues, **kwargs).indicateurs[critere]
        else:
            d = d.assign(_const_=1.0)
            meilleur = np.inf
        while candidates:
            scores = {}
            for c in candidates:
                try:
                    scores[c] = f(d, cible, retenues + [c], **kwargs).indicateurs[critere]
                except (np.linalg.LinAlgError, ValueError):
                    continue
            if not scores:
                break
            c_best = min(scores, key=scores.get)
            if scores[c_best] >= meilleur:
                break
            meilleur = scores[c_best]
            retenues.append(c_best)
            candidates.remove(c_best)
            histo.append({"etape": len(histo) + 1, "variable_ajoutee": c_best, critere: meilleur})
    if not retenues:
        raise ValueError("Aucune variable n'améliore le critère.")
    return f(d, cible, retenues, **kwargs), pd.DataFrame(histo)
