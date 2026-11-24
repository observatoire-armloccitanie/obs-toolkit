"""Validation croisée avec les bibliothèques de référence (si installées).

Ces tests sont ignorés si ``statsmodels`` ou ``lifelines`` ne sont pas
installés : ``pip install statsmodels lifelines`` pour les exécuter.
"""

import numpy as np
import pytest

from obskit import regression as rg
from obskit import survie as sv

EXPL = ["sexe", "qpv", "age", "niveau_formation", "permis_b"]


def test_ols_robuste_statsmodels(df):
    sm = pytest.importorskip("statsmodels.api")
    res = rg.regression_lineaire(df, "score_insertion", EXPL, erreurs="HC3")
    ref = sm.OLS(res.y, res.X).fit(cov_type="HC3")
    np.testing.assert_allclose(res.coefficients, ref.params, rtol=1e-8)
    np.testing.assert_allclose(res.tableau["erreur_std"], ref.bse, rtol=1e-6)
    assert res.indicateurs["r2"] == pytest.approx(ref.rsquared)


def test_cluster_statsmodels(df):
    sm = pytest.importorskip("statsmodels.api")
    res = rg.regression_lineaire(df, "score_insertion", ["age", "qpv"], grappes="mission_locale")
    codes = df["mission_locale"].astype("category").cat.codes.to_numpy()
    ref = sm.OLS(res.y, res.X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    np.testing.assert_allclose(res.tableau["erreur_std"], ref.bse, rtol=1e-6)


def test_logit_statsmodels(df):
    sm = pytest.importorskip("statsmodels.api")
    res = rg.regression_logistique(df, "acces_emploi", EXPL)
    ref = sm.Logit(res.y, res.X).fit(disp=0)
    np.testing.assert_allclose(res.coefficients, ref.params, rtol=1e-6)
    np.testing.assert_allclose(res.tableau["erreur_std"], ref.bse, rtol=1e-5)
    assert res.indicateurs["pseudo_r2_mcfadden"] == pytest.approx(ref.prsquared)
    assert res.indicateurs["aic"] == pytest.approx(ref.aic)


def test_poisson_statsmodels(df):
    sm = pytest.importorskip("statsmodels.api")
    res = rg.regression_poisson(df, "nb_entretiens", ["dispositif", "qpv"], exposition="duree_accompagnement_mois")
    ref = sm.GLM(res.y, res.X, family=sm.families.Poisson(), offset=res.offset).fit()
    np.testing.assert_allclose(res.coefficients, ref.params, rtol=1e-6)
    np.testing.assert_allclose(res.tableau["erreur_std"], ref.bse, rtol=1e-5)


def test_mnlogit_statsmodels(df):
    sm = pytest.importorskip("statsmodels.api")
    res = rg.regression_logistique_multinomiale(df, "situation_sortie", ["qpv", "age"], reference_cible="Autre")
    X = res.encodeur.transform(df)
    ref = sm.MNLogit(df["situation_sortie"], X).fit(disp=0, method="newton")
    np.testing.assert_allclose(res.coef, ref.params.to_numpy(), rtol=1e-5)


def test_km_lifelines(df):
    lifelines = pytest.importorskip("lifelines")
    km = sv.kaplan_meier(df, "duree_emploi_mois", "evenement_emploi")
    ref = lifelines.KaplanMeierFitter().fit(df["duree_emploi_mois"], df["evenement_emploi"])
    t = km.tables["Ensemble"]
    np.testing.assert_allclose(t["survie"], ref.survival_function_at_times(t["temps"]).to_numpy(), atol=1e-10)
    ci = ref.confidence_interval_survival_function_
    attendu = ci.iloc[:, 0].reindex(t["temps"].iloc[1:]).to_numpy()
    np.testing.assert_allclose(t["ic_bas"].to_numpy()[1:], attendu, atol=1e-6)


def test_logrank_lifelines(df):
    stats_ = pytest.importorskip("lifelines.statistics")
    lr = sv.test_logrank(df, "duree_emploi_mois", "evenement_emploi", "niveau_formation")
    ref = stats_.multivariate_logrank_test(df["duree_emploi_mois"], df["niveau_formation"], df["evenement_emploi"])
    assert lr["chi2"] == pytest.approx(ref.test_statistic, rel=1e-6)


def test_cox_lifelines(df):
    lifelines = pytest.importorskip("lifelines")
    res = sv.modele_cox(df, "duree_emploi_mois", "evenement_emploi", ["age", "qpv", "permis_b"])
    d = df[["duree_emploi_mois", "evenement_emploi", "age", "qpv", "permis_b"]].astype(float)
    ref = lifelines.CoxPHFitter().fit(d, "duree_emploi_mois", "evenement_emploi")
    np.testing.assert_allclose(res.coefficients.to_numpy(), ref.params_.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(res.tableau["erreur_std"].to_numpy(), ref.standard_errors_.to_numpy(), rtol=1e-4)
    assert res.indicateurs["concordance"] == pytest.approx(ref.concordance_index_, abs=1e-6)
