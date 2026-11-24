import numpy as np
import pandas as pd
import pytest
from scipy import stats

from obskit import inference as inf


def test_intervalle_wilson_et_clopper():
    lo, hi = inf.intervalle_proportion(45, 120)
    assert (round(lo, 3), round(hi, 3)) == (0.294, 0.464)
    ref = stats.binomtest(7, 40).proportion_ci(method="exact")
    lo, hi = inf.intervalle_proportion(7, 40, methode="clopper-pearson")
    assert lo == pytest.approx(ref.low) and hi == pytest.approx(ref.high)
    ref = stats.binomtest(7, 40).proportion_ci(method="wilson")
    lo, hi = inf.intervalle_proportion(7, 40)
    assert lo == pytest.approx(ref.low) and hi == pytest.approx(ref.high)
    assert inf.intervalle_proportion(0, 30)[0] == 0.0


def test_intervalle_moyenne(rng):
    x = rng.normal(10, 2, 100)
    m, lo, hi = inf.intervalle_moyenne(x)
    ref = stats.t.interval(0.95, 99, loc=x.mean(), scale=stats.sem(x))
    assert lo == pytest.approx(ref[0]) and hi == pytest.approx(ref[1])
    m2, lo2, hi2 = inf.intervalle_moyenne(x, poids=np.ones(100))
    assert m2 == pytest.approx(m) and lo2 == pytest.approx(lo)


def test_corriger_pvaleurs():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(inf.corriger_pvaleurs(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(inf.corriger_pvaleurs(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(inf.corriger_pvaleurs(p, "fdr_bh"), [0.02, 0.04, 0.04, 0.02])
    out = inf.corriger_pvaleurs([0.01, np.nan], "holm")
    assert np.isnan(out[1]) and out[0] == 0.01


def test_test_independance(df):
    t = inf.test_independance(df, "niveau_formation", "situation_sortie")
    tab = pd.crosstab(df.niveau_formation, df.situation_sortie)
    ref = stats.chi2_contingency(tab, correction=False)
    assert t["statistique"] == pytest.approx(ref[0]) and t["ddl"] == ref[2]
    assert 0 <= t["v_cramer"] <= 1


def test_fisher_petits_effectifs():
    d = pd.DataFrame({"a": ["x"] * 6 + ["y"] * 6, "b": ["o"] * 5 + ["n"] * 1 + ["o"] * 1 + ["n"] * 5})
    t = inf.test_independance(d, "a", "b")
    assert t["test"] == "fisher_exact"
    assert t["p_valeur"] == pytest.approx(stats.fisher_exact([[1, 5], [5, 1]])[1])


def test_v_cramer_extremes(rng):
    x = rng.choice(list("abc"), 300)
    assert inf.v_cramer(x, x, correction=False) == pytest.approx(1.0)
    y = rng.choice(list("uv"), 300)
    assert inf.v_cramer(x, y) < 0.15


def test_comparer_groupes(df, rng):
    r = inf.comparer_groupes(df, "score_insertion", "sexe")
    assert r["test"] == "t de Welch"
    a = df.loc[df.sexe == "Femme", "score_insertion"]
    b = df.loc[df.sexe == "Homme", "score_insertion"]
    assert r["p_valeur"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    r = inf.comparer_groupes(df, "score_insertion", "niveau_formation")
    assert r["test"] == "ANOVA de Welch" and r["p_valeur"] < 1e-10
    petit = pd.DataFrame({"v": rng.exponential(1, 24), "g": list("ab") * 12})
    assert inf.comparer_groupes(petit, "v", "g", parametrique=False)["test"] == "Mann-Whitney"
    r = inf.comparer_groupes(df, "score_insertion", "dispositif", parametrique=False)
    assert r["test"] == "Kruskal-Wallis"


def test_anova_welch_vs_reference(rng):
    # comparaison avec scipy (ANOVA classique) quand variances et effectifs égaux
    g = [rng.normal(m, 1, 200) for m in (0, 0.1, 0.3)]
    d = pd.DataFrame({"v": np.concatenate(g), "g": np.repeat(list("abc"), 200)})
    r = inf.comparer_groupes(d, "v", "g", parametrique=True)
    assert r["statistique"] == pytest.approx(stats.f_oneway(*g).statistic, rel=0.05)


def test_comparer_proportions():
    r = inf.comparer_proportions(60, 100, 45, 100)
    tab = np.array([[60, 40], [45, 55]])
    assert r["z"] ** 2 == pytest.approx(stats.chi2_contingency(tab, correction=False)[0])
    assert r["difference"] == pytest.approx(0.15)


def test_taille_echantillon():
    assert inf.taille_echantillon_proportion(0.05) == 385
    assert inf.taille_echantillon_proportion(0.05, population=1000) == 278


def test_normalite(rng):
    r = inf.test_normalite(rng.normal(size=500))
    assert r["shapiro_p"] > 0.001
    assert inf.test_normalite(rng.exponential(size=500))["shapiro_p"] < 0.001
