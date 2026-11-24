import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

from obskit import regression as rg
from obskit.donnees_exemple import PARAMETRES_VRAIS

EXPL = ["sexe", "qpv", "age", "niveau_formation", "permis_b"]
REF = {"sexe": "Femme", "niveau_formation": "Niveau 3 (CAP-BEP)"}


def test_mco_identique_a_lstsq(rng):
    n = 500
    X = rng.normal(size=(n, 3))
    y = 1 + X @ [0.5, -1.0, 2.0] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(y=y)
    res = rg.regression_lineaire(df, "y", ["a", "b", "c"])
    Z = np.column_stack([np.ones(n), X])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    np.testing.assert_allclose(res.coefficients.to_numpy(), b, rtol=1e-10)
    resid = y - Z @ b
    sigma2 = resid @ resid / (n - 4)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Z.T @ Z)))
    np.testing.assert_allclose(res.tableau["erreur_std"].to_numpy(), se, rtol=1e-8)
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    assert res.indicateurs["r2"] == pytest.approx(r2)


def test_erreurs_robustes_hc1_et_hc3(rng):
    n = 400
    x = rng.normal(size=n)
    y = 1 + 2 * x + rng.normal(size=n) * (1 + np.abs(x))
    df = pd.DataFrame({"x": x, "y": y})
    Z = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ b
    bread = np.linalg.inv(Z.T @ Z)
    hc0 = bread @ (Z.T * e ** 2) @ Z @ bread
    res1 = rg.regression_lineaire(df, "y", ["x"], erreurs="HC1")
    np.testing.assert_allclose(res1.tableau["erreur_std"] ** 2, np.diag(hc0) * n / (n - 2), rtol=1e-8)
    h = np.einsum("ij,jk,ik->i", Z, bread, Z)
    hc3 = bread @ (Z.T * (e / (1 - h)) ** 2) @ Z @ bread
    res3 = rg.regression_lineaire(df, "y", ["x"], erreurs="HC3")
    np.testing.assert_allclose(res3.tableau["erreur_std"] ** 2, np.diag(hc3), rtol=1e-8)


def test_erreurs_cluster(df):
    res = rg.regression_lineaire(df, "score_insertion", ["sexe", "age"], grappes="mission_locale")
    assert res.ajustement.type_cov == "cluster"
    assert np.all(res.tableau["erreur_std"] > 0)


def test_logistique_identique_sklearn(df):
    sk = pytest.importorskip("sklearn.linear_model")
    res = rg.regression_logistique(df, "acces_emploi", EXPL, reference=REF)
    X = res.X.drop(columns="const").to_numpy()
    m = sk.LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(X, res.y)
    np.testing.assert_allclose(res.coefficients.iloc[1:], m.coef_[0], atol=1e-4)
    assert res.coefficients.iloc[0] == pytest.approx(m.intercept_[0], abs=1e-4)


def test_logistique_retrouve_vrais_parametres(df_grand):
    res = rg.regression_logistique(df_grand, "acces_emploi", EXPL + ["dispositif"],
                                   reference={**REF, "dispositif": "PACEA"})
    vrais = PARAMETRES_VRAIS["logistique"]
    for terme, val in vrais.items():
        est, se = res.tableau.loc[terme, ["coef", "erreur_std"]]
        assert abs(est - val) < 4 * se, terme


def test_lineaire_retrouve_vrais_parametres(df_grand):
    res = rg.regression_lineaire(df_grand, "score_insertion", EXPL, reference=REF)
    for terme, val in PARAMETRES_VRAIS["lineaire"].items():
        if terme == "sigma":
            assert res.indicateurs["sigma"] == pytest.approx(val, rel=0.05)
            continue
        est, se = res.tableau.loc[terme, ["coef", "erreur_std"]]
        assert abs(est - val) < 4 * se, terme


def test_poisson_exposition(df_grand):
    res = rg.regression_poisson(df_grand, "nb_entretiens", ["dispositif", "qpv"],
                                exposition="duree_accompagnement_mois", reference={"dispositif": "PACEA"})
    for terme, val in PARAMETRES_VRAIS["poisson"].items():
        est, se = res.tableau.loc[terme, ["coef", "erreur_std"]]
        assert abs(est - val) < 4 * se, terme
    assert "irr" in res.tableau.columns


def test_poisson_identique_sklearn(df):
    sk = pytest.importorskip("sklearn.linear_model")
    res = rg.regression_poisson(df, "nb_entretiens", ["dispositif", "qpv", "age"])
    X = res.X.drop(columns="const").to_numpy()
    m = sk.PoissonRegressor(alpha=0, max_iter=1000, tol=1e-10).fit(X, res.y)
    np.testing.assert_allclose(res.coefficients.iloc[1:], m.coef_, atol=1e-4)


def test_ponderation_equivaut_duplication(rng):
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    df["y"] = (rng.random(n) < special.expit(0.5 * df.x)).astype(int)
    df["w"] = rng.integers(1, 4, n)
    dup = df.loc[df.index.repeat(df.w)]
    r1 = rg.regression_logistique(df, "y", ["x", "g"], poids="w")
    r2 = rg.regression_logistique(dup, "y", ["x", "g"])
    np.testing.assert_allclose(r1.coefficients, r2.coefficients, atol=1e-8)


def test_valeurs_manquantes_exclues():
    from obskit.donnees_exemple import generer_jeunes

    d = generer_jeunes(800, graine=3, taux_manquants=0.05)
    with pytest.warns(UserWarning, match="exclue"):
        res = rg.regression_logistique(d, "acces_emploi", ["niveau_formation", "permis_b", "age"])
    assert res.indicateurs["n"] + res.indicateurs["n_exclus"] == len(d)


def test_cible_non_binaire_et_modalite_positive(df):
    with pytest.raises(ValueError):
        rg.regression_logistique(df, "situation_sortie", ["age"])
    res = rg.regression_logistique(df, "situation_sortie", ["age", "qpv"], modalite_positive="Emploi")
    assert res.modalite_positive == "Emploi"
    assert set(np.unique(res.y)) == {0.0, 1.0}


def test_prediction_coherente(df):
    res = rg.regression_logistique(df, "acces_emploi", EXPL)
    np.testing.assert_allclose(res.predire(df).to_numpy(), res.ajustement.mu, rtol=1e-10)
    lin = rg.regression_lineaire(df, "score_insertion", EXPL)
    np.testing.assert_allclose(lin.predire(df).to_numpy(), lin.ajustement.mu, rtol=1e-10)


def test_effets_marginaux(df):
    lin = rg.regression_lineaire(df, "score_insertion", EXPL)
    ame = rg.effets_marginaux_moyens(lin)
    age = ame.loc[ame.variable == "age", "ame"].iloc[0]
    assert age == pytest.approx(lin.coefficients["age"], rel=1e-5)
    se_age = ame.loc[ame.variable == "age", "erreur_std"].iloc[0]
    assert se_age == pytest.approx(lin.tableau.loc["age", "erreur_std"], rel=1e-3)
    log = rg.regression_logistique(df, "acces_emploi", EXPL, reference=REF)
    ame = rg.effets_marginaux_moyens(log, ["permis_b"])
    d1, d0 = df.copy(), df.copy()
    d1["permis_b"], d0["permis_b"] = True, False
    attendu = (log.predire(d1) - log.predire(d0)).mean()
    assert ame["ame"].iloc[0] == pytest.approx(attendu, rel=1e-6)


def test_tableau_complet_contient_references(df):
    res = rg.regression_logistique(df, "acces_emploi", EXPL, reference=REF)
    tc = res.tableau_complet()
    ref = tc[(tc.variable == "niveau_formation") & (tc.modalite == "Niveau 3 (CAP-BEP)")]
    assert ref["signif"].iloc[0] == "Réf." and ref["odds_ratio"].iloc[0] == 1.0
    assert isinstance(res.resume(), str)


def test_tests_wald_et_rv(df):
    complet = rg.regression_logistique(df, "acces_emploi", EXPL)
    restreint = rg.regression_logistique(df, "acces_emploi", ["sexe", "qpv", "age", "permis_b"])
    rv = rg.test_rapport_vraisemblance(restreint, complet)
    assert rv["ddl"] == 3 and rv["p_valeur"] < 0.001
    tg = rg.test_global_variables(complet)
    assert tg.set_index("variable").loc["niveau_formation", "ddl"] == 3
    comp = rg.comparer_modeles({"restreint": restreint, "complet": complet})
    assert comp.loc["complet", "aic"] < comp.loc["restreint", "aic"]


def test_auc(rng):
    sk = pytest.importorskip("sklearn.metrics")
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y
    assert rg.auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    w = rng.uniform(0.5, 2, 500)
    assert rg.auc(y, s, poids=w) == pytest.approx(sk.roc_auc_score(y, s, sample_weight=w))


def test_performance_et_hosmer(df):
    res = rg.regression_logistique(df, "acces_emploi", EXPL)
    perf = rg.performance_classification(res)
    assert perf["matrice_confusion"].to_numpy().sum() == res.indicateurs["n"]
    hl = rg.test_hosmer_lemeshow(res)
    assert hl["p_valeur"] > 0.01  # modèle bien spécifié


def test_vif(df):
    v = rg.vif(df, EXPL)
    assert (v["gvif"] < 1.5).all()
    d = df.assign(age2=df["age"] * 2 + np.random.default_rng(0).normal(0, 0.1, len(df)))
    assert rg.vif(d, ["age", "age2"])["gvif"].min() > 50


def test_selection_pas_a_pas(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, histo = rg.selection_pas_a_pas(df, "acces_emploi", EXPL + ["zrr", "departement"])
    assert "niveau_formation" in res.explicatives
    assert histo["aic"].is_monotonic_decreasing


def test_multinomiale_identique_sklearn(df):
    sk = pytest.importorskip("sklearn.linear_model")
    res = rg.regression_logistique_multinomiale(df, "situation_sortie", ["qpv", "permis_b", "age"],
                                                reference_cible="Autre")
    X = res.encodeur.transform(df).drop(columns="const").to_numpy()
    m = sk.LogisticRegression(penalty=None, max_iter=10000, tol=1e-10).fit(X, df["situation_sortie"])
    classes = list(m.classes_)
    ref = classes.index("Autre")
    for mod in ["Emploi", "Formation", "Sans solution"]:
        j = classes.index(mod)
        attendu = m.coef_[j] - m.coef_[ref]
        np.testing.assert_allclose(res.tableau.loc[mod, "coef"].iloc[1:], attendu, atol=1e-3)
    P = res.predire(df.head(5))
    np.testing.assert_allclose(P.sum(axis=1), 1)
