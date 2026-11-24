import numpy as np
import pandas as pd
import pytest
from scipy import stats

from obskit import factoriel as fa

ACTIVES = ["sexe", "qpv", "niveau_formation", "permis_b", "dispositif"]


def test_acp(df):
    vars_ = ["age", "score_insertion", "nb_entretiens", "duree_accompagnement_mois"]
    r = fa.acp(df, vars_)
    assert r.valeurs_propres["valeur_propre"].sum() == pytest.approx(4)
    ev = np.linalg.eigvalsh(np.corrcoef(df[vars_].to_numpy(dtype=float), rowvar=False))[::-1]
    np.testing.assert_allclose(r.valeurs_propres["valeur_propre"], ev, rtol=1e-8)
    np.testing.assert_allclose(r.variables["contrib"].sum(), 100)
    np.testing.assert_allclose(r.individus["contrib"].sum(), 100)
    # coordonnées des variables = corrélations avec les composantes
    c = np.corrcoef(df["age"], r.individus["coord"]["Dim.1"])[0, 1]
    assert r.variables["coord"].loc["age", "Dim.1"] == pytest.approx(c)
    sup = fa.acp(df, vars_[:3], variables_sup=["duree_accompagnement_mois"])
    assert "variables" in sup.supplementaires


def test_afc(df):
    tab = pd.crosstab(df.niveau_formation, df.situation_sortie)
    r = fa.afc(tab)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    assert r.valeurs_propres["valeur_propre"].sum() == pytest.approx(chi2 / tab.to_numpy().sum())
    assert len(r.valeurs_propres) == min(tab.shape) - 1
    np.testing.assert_allclose(r.individus["contrib"].sum(), 100)
    np.testing.assert_allclose(r.variables["contrib"].sum(), 100)


def test_acm(df):
    r = fa.acm(df, ACTIVES, variables_sup=["situation_sortie"], correction_benzecri=True)
    J = sum(df[v].nunique() for v in ACTIVES)
    Q = len(ACTIVES)
    assert r.valeurs_propres["valeur_propre"].sum() == pytest.approx((J - Q) / Q)
    np.testing.assert_allclose(r.variables["contrib"].sum(), 100)
    assert "pct_benzecri" in r.valeurs_propres
    assert r.supplementaires["modalites"]["coord"].shape[0] == df.situation_sortie.nunique()
    # eta² d'une variable = moyenne pondérée des carrés des coordonnées individus par modalité / vp
    assert (r.variables["eta2"].to_numpy() <= 1 + 1e-9).all()
    # somme des eta² sur les variables = Q * valeur propre
    np.testing.assert_allclose(r.variables["eta2"].sum().to_numpy(),
                               Q * r.valeurs_propres["valeur_propre"].iloc[:5].to_numpy(), rtol=1e-8)


def test_cah_et_caracterisation(df):
    r = fa.acm(df, ACTIVES)
    cl = fa.classification(r.individus["coord"].iloc[:, :3], n_classes=4)
    assert cl["classes"].nunique() == 4
    auto = fa.classification(r.individus["coord"].iloc[:, :3])
    assert 3 <= auto["n_classes"] <= 10
    car = fa.caracteriser_classes(df, cl["classes"], ACTIVES)
    assert set(car["classe"]) <= {1, 2, 3, 4}
    assert (car["p_valeur"] < 0.05).all()
