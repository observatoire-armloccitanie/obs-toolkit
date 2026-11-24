import numpy as np
import pandas as pd
import pytest

from obskit import causalite as ca


@pytest.fixture(scope="module")
def donnees_causales():
    rng = np.random.default_rng(11)
    n = 5000
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 + 0.7 * (x2 == "b") - 0.3)))).astype(int)
    y = 2.0 * t + 1.5 * x1 + 1.0 * (x2 == "b") + rng.normal(size=n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.8 * t + 0.6 * x1)))).astype(int)
    return pd.DataFrame({"t": t, "y": y, "yb": yb, "x1": x1, "x2": x2})


@pytest.mark.parametrize("methode", ["regression", "ipw", "aipw", "appariement"])
def test_effet_retrouve(donnees_causales, methode):
    d = donnees_causales
    naif = d.loc[d.t == 1, "y"].mean() - d.loc[d.t == 0, "y"].mean()
    assert naif > 2.5  # biais de sélection
    r = ca.estimer_effet(d, "t", "y", ["x1", "x2"], methode=methode)
    assert abs(r.effet - 2.0) < max(4 * r.erreur_std, 0.1)
    assert r.ic_bas < r.effet < r.ic_haut
    if r.equilibre_apres is not None:
        assert r.equilibre_apres["smd"].abs().max() < r.equilibre_avant["smd"].abs().max()
    assert isinstance(r.resume(), str)


def test_att_aipw_et_resultat_binaire(donnees_causales):
    r = ca.estimer_effet(donnees_causales, "t", "yb", ["x1", "x2"], methode="aipw", estimand="ATT")
    assert r.details["resultat_binaire"] and -1 < r.effet < 1 and r.effet > 0


def test_bootstrap(donnees_causales):
    d = donnees_causales.sample(1500, random_state=1)
    r = ca.estimer_effet(d, "t", "y", ["x1", "x2"], methode="ipw", bootstrap=30)
    assert len(r.details["bootstrap"]) == 30 and r.erreur_std > 0


def test_equilibre_et_poids(donnees_causales):
    d = donnees_causales
    eq = ca.equilibre_covariables(d, "t", ["x1", "x2"])
    assert not eq.set_index("covariable").loc["x1", "equilibre"]
    ps = ca.score_propension(d, "t", ["x1", "x2"])
    assert ps.between(0, 1).all()
    w = ca.poids_ipw(ps, d.t, "ATT")
    assert np.all(w[d.t.to_numpy() == 1] == 1)
    with pytest.raises(ValueError):
        ca.poids_ipw(ps, d.t, "XYZ")


@pytest.mark.filterwarnings("ignore:.*sans témoin")
def test_appariement(donnees_causales):
    m = ca.appariement(donnees_causales, "t", ["x1", "x2"], caliper=0.2)
    tr = m[m.t == 1]
    assert (tr["_poids_appariement"] == 1).all()
    assert m.loc[m.t == 0, "_poids_appariement"].sum() == pytest.approx(len(tr))
    m2 = ca.appariement(donnees_causales, "t", ["x1", "x2"], remise=True, ratio=2)
    assert m2.loc[m2.t == 0, "_poids_appariement"].sum() == pytest.approx((m2.t == 1).sum())


def test_traitement_non_binaire(donnees_causales):
    with pytest.raises(ValueError):
        ca.estimer_effet(donnees_causales, "x2", "y", ["x1"])


def test_difference_en_differences(rng):
    n = 4000
    g, p = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = 1 + 0.5 * g + 0.3 * p + 1.2 * g * p + rng.normal(size=n)
    d = pd.DataFrame({"y": y, "g": g, "p": p, "terr": rng.integers(0, 40, n)})
    r = ca.difference_en_differences(d, "y", "g", "p", grappes="terr")
    assert abs(r["effet"] - 1.2) < 4 * r["erreur_std"]
    m = r["moyennes"]
    manuel = ((m.loc["Traités", "Après"] - m.loc["Traités", "Avant"])
              - (m.loc["Témoins", "Après"] - m.loc["Témoins", "Avant"]))
    assert r["effet"] == pytest.approx(manuel)


def test_e_value():
    assert ca.e_value(1.8)["e_value"] == pytest.approx(3.0)
    assert ca.e_value(1 / 1.8)["e_value"] == pytest.approx(3.0)
    assert ca.e_value(1.8, 0.9)["e_value_ic"] == 1.0
    assert ca.e_value(4.0, type_mesure="OR")["e_value"] == pytest.approx(2 + np.sqrt(2))
