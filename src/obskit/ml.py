"""Modélisation prédictive (apprentissage automatique) avec scikit-learn.

Complément aux modèles explicatifs de :mod:`obskit.regression` : ici l'objectif
est de **bien prédire** (repérer les jeunes à risque de décrochage, prévoir
des volumes…) plutôt que d'interpréter des coefficients. Les performances sont
toujours évaluées sur des données non vues (échantillon test + validation croisée).

Nécessite scikit-learn : ``pip install obskit[ml]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import est_categorielle, explicatives_par_defaut, verifier_colonnes

__all__ = ["modele_predictif", "ResultatML", "segmentation_kmeans"]


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("Ce module nécessite scikit-learn : pip install scikit-learn") from e


@dataclass
class ResultatML:
    """Résultat de :func:`modele_predictif`.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Prétraitement + modèle, réentraîné sur l'échantillon d'apprentissage.
    metriques_test : dict
        Performances sur l'échantillon test.
    validation_croisee : DataFrame
        Scores par pli de validation croisée (apprentissage).
    importance : DataFrame
        Importance par permutation des variables (baisse de performance sur le
        test quand la variable est brouillée), avec écart-type.
    """

    tache: str
    algorithme: str
    cible: str
    explicatives: list[str]
    pipeline: object = field(repr=False)
    metriques_test: dict = field(default_factory=dict)
    validation_croisee: pd.DataFrame | None = field(default=None, repr=False)
    importance: pd.DataFrame | None = field(default=None, repr=False)
    classes: list | None = None

    def predire(self, df: pd.DataFrame) -> pd.Series:
        """Prédictions (classe ou valeur) pour de nouvelles données."""
        return pd.Series(self.pipeline.predict(df[self.explicatives]), index=df.index, name="prediction")

    def predire_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probabilités par classe (classification uniquement)."""
        if self.tache != "classification":
            raise ValueError("Réservé à la classification.")
        return pd.DataFrame(self.pipeline.predict_proba(df[self.explicatives]), index=df.index,
                            columns=self.classes)


def _pipeline(X: pd.DataFrame, tache: str, algorithme: str, graine: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cat = [c for c in X.columns if est_categorielle(X[c])]
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                            sparse_output=False)), cat),
    ], verbose_feature_names_out=False)
    modeles = {
        ("classification", "lineaire"): LogisticRegression(max_iter=2000),
        ("classification", "foret_aleatoire"): RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                      random_state=graine, n_jobs=-1),
        ("classification", "gradient_boosting"): HistGradientBoostingClassifier(random_state=graine),
        ("classification", "arbre"): DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=graine),
        ("regression", "lineaire"): LinearRegression(),
        ("regression", "foret_aleatoire"): RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                                 random_state=graine, n_jobs=-1),
        ("regression", "gradient_boosting"): HistGradientBoostingRegressor(random_state=graine),
        ("regression", "arbre"): DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=graine),
    }
    if (tache, algorithme) not in modeles:
        raise ValueError("algorithme : 'lineaire', 'foret_aleatoire', 'gradient_boosting' ou 'arbre'.")
    return Pipeline([("preparation", prep), ("modele", modeles[(tache, algorithme)])])


def modele_predictif(
    df: pd.DataFrame,
    cible: str,
    explicatives: Sequence[str] | None = None,
    tache: str = "auto",
    algorithme: str = "gradient_boosting",
    part_test: float = 0.25,
    cv: int = 5,
    graine: int = 42,
    importance: bool = True,
) -> ResultatML:
    """Entraîne et évalue un modèle prédictif sur une cible quelconque.

    Parameters
    ----------
    cible : str
        Variable à prédire (catégorielle → classification, numérique → régression).
    explicatives : list of str, optional
        Prédicteurs (défaut : toutes les autres colonnes ; exclure identifiants et dates !).
    tache : {"auto", "classification", "regression"}
    algorithme : {"gradient_boosting", "foret_aleatoire", "lineaire", "arbre"}
        ``"arbre"`` (profondeur 4) donne des règles lisibles ; ``"gradient_boosting"``
        est souvent le plus performant et gère nativement les manquants.
    part_test : float
        Part des données réservée au test final.
    cv : int
        Nombre de plis de validation croisée sur l'apprentissage.

    Returns
    -------
    ResultatML
    """
    _sklearn()
    from sklearn import metrics
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_validate, train_test_split

    explicatives = explicatives_par_defaut(df, cible, explicatives)
    verifier_colonnes(df, [cible, *explicatives])
    d = df[[cible, *explicatives]].dropna(subset=[cible])
    X, y = d[explicatives], d[cible]
    if tache == "auto":
        tache = "classification" if (est_categorielle(y) or y.nunique() <= 2) else "regression"
    if tache == "classification":
        y = y.astype(str)
    strat = y if tache == "classification" else None
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=part_test, random_state=graine, stratify=strat)
    pipe = _pipeline(X, tache, algorithme, graine)
    if tache == "classification":
        if y.nunique() == 2:
            scoring = ["roc_auc", "accuracy", "balanced_accuracy"]
        else:
            scoring = ["accuracy", "balanced_accuracy", "f1_macro"]
    else:
        scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    cvres = cross_validate(pipe, Xa, ya, cv=cv, scoring=scoring)
    vc = pd.DataFrame({k.replace("test_", ""): v for k, v in cvres.items() if k.startswith("test_")})
    pipe.fit(Xa, ya)
    pred = pipe.predict(Xt)
    if tache == "classification":
        met = {"exactitude": metrics.accuracy_score(yt, pred),
               "exactitude_equilibree": metrics.balanced_accuracy_score(yt, pred),
               "f1_macro": metrics.f1_score(yt, pred, average="macro"),
               "matrice_confusion": pd.DataFrame(metrics.confusion_matrix(yt, pred, labels=pipe.classes_),
                                                 index=pipe.classes_, columns=pipe.classes_),
               "rapport": metrics.classification_report(yt, pred, output_dict=True, zero_division=0)}
        if len(pipe.classes_) == 2:
            proba = pipe.predict_proba(Xt)[:, 1]
            met["auc"] = metrics.roc_auc_score(yt, proba)
            met["brier"] = metrics.brier_score_loss((yt == pipe.classes_[1]).astype(int), proba)
        score_perm = "roc_auc" if len(pipe.classes_) == 2 else "balanced_accuracy"
    else:
        met = {"r2": metrics.r2_score(yt, pred), "mae": metrics.mean_absolute_error(yt, pred),
               "rmse": float(np.sqrt(metrics.mean_squared_error(yt, pred)))}
        score_perm = "r2"
    imp = None
    if importance:
        pi = permutation_importance(pipe, Xt, yt, n_repeats=10, random_state=graine, scoring=score_perm)
        imp = pd.DataFrame({"variable": explicatives, "importance": pi.importances_mean,
                            "ecart_type": pi.importances_std}).sort_values("importance", ascending=False)
        imp = imp.reset_index(drop=True)
    return ResultatML(tache=tache, algorithme=algorithme, cible=cible, explicatives=explicatives,
                      pipeline=pipe, metriques_test=met, validation_croisee=vc, importance=imp,
                      classes=list(pipe.classes_) if tache == "classification" else None)


def segmentation_kmeans(
    df: pd.DataFrame, variables: Sequence[str], k: int | None = None, k_max: int = 8, graine: int = 42,
) -> dict:
    """Segmentation en k groupes (k-means) sur variables numériques et/ou catégorielles.

    Les variables sont standardisées (numériques) ou codées en indicatrices
    (catégorielles). Si ``k`` est ``None``, il est choisi par le score de
    silhouette entre 2 et ``k_max``. Pour des variables uniquement
    qualitatives, préférer ACM + CAH (:mod:`obskit.factoriel`).

    Returns
    -------
    dict : ``classes`` (Series), ``k``, ``silhouettes`` (DataFrame), ``profils``
    (moyennes des variables numériques et parts des modalités par classe).
    """
    _sklearn()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    variables = list(variables)
    verifier_colonnes(df, variables)
    d = df[variables].dropna()
    cat = [c for c in variables if est_categorielle(d[c])]
    num = [c for c in variables if c not in cat]
    parts = []
    if num:
        z = (d[num] - d[num].mean()) / d[num].std(ddof=0)
        parts.append(z)
    if cat:
        parts.append(pd.get_dummies(d[cat].astype(str)).astype(float))
    X = pd.concat(parts, axis=1).to_numpy()
    sil = []
    if k is None:
        echant = np.random.default_rng(graine).choice(len(X), min(len(X), 5000), replace=False)
        for kk in range(2, k_max + 1):
            lab = KMeans(kk, n_init=10, random_state=graine).fit_predict(X)
            sil.append({"k": kk, "silhouette": silhouette_score(X[echant], lab[echant])})
        k = int(max(sil, key=lambda s: s["silhouette"])["k"])
    km = KMeans(k, n_init=20, random_state=graine).fit(X)
    classes = pd.Series(km.labels_ + 1, index=d.index, name="classe")
    profils = []
    if num:
        profils.append(d[num].groupby(classes).mean())
    if cat:
        profils.append(pd.get_dummies(d[cat].astype(str)).astype(float).groupby(classes).mean() * 100)
    prof = pd.concat(profils, axis=1)
    prof.insert(0, "effectif", classes.value_counts().sort_index())
    return {"classes": classes, "k": k, "silhouettes": pd.DataFrame(sil), "profils": prof}
