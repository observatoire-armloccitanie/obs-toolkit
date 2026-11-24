import numpy as np
import pandas as pd
import pytest
from scipy import stats

from obskit import correlation as cr


def test_matrice_correlation(df):
    m = cr.matrice_correlation(df, ["age", "score_insertion", "nb_entretiens"], correction=None)
    r, p = stats.pearsonr(df.age, df.score_insertion)
    assert m["r"].loc["age", "score_insertion"] == pytest.approx(r)
    assert m["p_valeur"].loc["age", "score_insertion"] == pytest.approx(p)
    np.testing.assert_allclose(m["r"], m["r"].T)
    long = cr.matrice_correlation(df, ["age", "score_insertion", "nb_entretiens"], methode="spearman", format_long=True)
    assert len(long) == 3 and long["r"].abs().is_monotonic_decreasing


def test_correlation_partielle(rng):
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)
    d = pd.DataFrame({"x": x, "y": y, "z": z})
    assert np.corrcoef(x, y)[0, 1] > 0.3
    assert abs(cr.correlation_partielle(d, "x", "y", ["z"])["r_partiel"]) < 0.06


def test_rapport_correlation(rng):
    g = rng.choice(list("abc"), 300)
    v = pd.Series(g).map({"a": 1.0, "b": 2.0, "c": 3.0})
    assert cr.rapport_correlation(g, v) == pytest.approx(1.0)
    assert cr.rapport_correlation(g, rng.normal(size=300)) < 0.2


def test_matrice_association_et_cible(df):
    m = cr.matrice_association(df, ["sexe", "age", "niveau_formation", "score_insertion"])
    assert m.shape == (4, 4) and ((m.to_numpy() >= 0) & (m.to_numpy() <= 1 + 1e-12)).all()
    a = cr.associations_avec_cible(df, "acces_emploi", ["sexe", "age", "niveau_formation", "departement", "permis_b"])
    assert a["variable"].iloc[0] == "niveau_formation"
    assert a["association"].is_monotonic_decreasing
    a2 = cr.associations_avec_cible(df, "situation_sortie", ["qpv", "age"])
    assert set(a2["test"]) <= {"chi2", "Kruskal-Wallis", "fisher_exact"}
