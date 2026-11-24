import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from obskit import survie as sv
from obskit.donnees_exemple import PARAMETRES_VRAIS


@pytest.fixture
def petites(rng):
    n = 200
    x = rng.normal(size=n)
    g = rng.choice(["A", "B"], n)
    t = rng.exponential(1 / np.exp(0.7 * x + 0.5 * (g == "B")))
    c = rng.exponential(2.0, n)
    return pd.DataFrame({"duree": np.minimum(t, c), "evt": (t <= c).astype(int), "x": x, "g": g})


def test_km_identique_scipy(petites):
    km = sv.kaplan_meier(petites, "duree", "evt")
    tab = km.tables["Ensemble"]
    cd = stats.CensoredData(uncensored=petites.duree[petites.evt == 1].to_numpy(),
                            right=petites.duree[petites.evt == 0].to_numpy())
    ref = stats.ecdf(cd).sf
    np.testing.assert_allclose(tab["survie"].to_numpy()[1:], ref.evaluate(tab["temps"].to_numpy()[1:]), atol=1e-12)


def test_km_exemple_manuel():
    # durées 1,2,2+,3,4+ : S(1)=4/5, S(2)=4/5*3/4=0.6, S(3)=0.6*1/2=0.3
    d = pd.DataFrame({"t": [1, 2, 2, 3, 4], "e": [1, 1, 0, 1, 0]})
    tab = sv.kaplan_meier(d, "t", "e").tables["Ensemble"].set_index("temps")
    assert tab.loc[1, "survie"] == pytest.approx(0.8)
    assert tab.loc[2, "survie"] == pytest.approx(0.6)
    assert tab.loc[3, "survie"] == pytest.approx(0.3)
    km = sv.kaplan_meier(d, "t", "e")
    assert km.medianes["mediane"].iloc[0] == 3
    assert km.rmst(4)["rmst"].iloc[0] == pytest.approx(1 + 0.8 + 0.6 + 0.3)


def test_logrank_identique_scipy(petites):
    lr = sv.test_logrank(petites, "duree", "evt", "g")
    a, b = petites[petites.g == "A"], petites[petites.g == "B"]
    mk = lambda d: stats.CensoredData(uncensored=d.duree[d.evt == 1].to_numpy(),  # noqa: E731
                                      right=d.duree[d.evt == 0].to_numpy())
    ref = stats.logrank(mk(a), mk(b))
    assert lr["chi2"] == pytest.approx(ref.statistic ** 2, rel=1e-8)
    assert lr["p_valeur"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_logrank_k_groupes(df):
    lr = sv.test_logrank(df, "duree_emploi_mois", "evenement_emploi", "niveau_formation")
    assert lr["ddl"] == 3 and lr["p_valeur"] < 1e-6
    assert lr["observes_attendus"]["observes"].sum() == pytest.approx(lr["observes_attendus"]["attendus"].sum())


def _breslow_nll(beta, X, T, E):
    eta = X @ beta
    ll = 0.0
    for i in np.flatnonzero(E == 1):
        ll += eta[i] - np.log(np.exp(eta[T >= T[i]]).sum())
    return -ll


def test_cox_identique_optimisation_directe(petites):
    d = petites.assign(gB=(petites.g == "B").astype(float))
    res = sv.modele_cox(d, "duree", "evt", ["x", "gB"])
    X = d[["x", "gB"]].to_numpy()
    opt = optimize.minimize(_breslow_nll, np.zeros(2), args=(X, d.duree.to_numpy(), d.evt.to_numpy()),
                            method="BFGS", options={"gtol": 1e-8})
    np.testing.assert_allclose(res.coefficients, opt.x, atol=1e-4)
    # sans ex aequo, Efron = Breslow
    res_b = sv.modele_cox(d, "duree", "evt", ["x", "gB"], methode="breslow")
    np.testing.assert_allclose(res.tableau["coef"], res_b.tableau["coef"], atol=1e-8)


def test_cox_efron_differe_avec_ex_aequo(df):
    d = df.assign(duree_mois=np.ceil(df["duree_emploi_mois"]))
    e = sv.modele_cox(d, "duree_mois", "evenement_emploi", ["qpv", "age"], methode="efron")
    b = sv.modele_cox(d, "duree_mois", "evenement_emploi", ["qpv", "age"], methode="breslow")
    assert not np.allclose(e.tableau["coef"], b.tableau["coef"])
    # Breslow atténue les coefficients en présence de nombreux ex aequo
    assert abs(b.tableau.loc["age", "coef"]) < abs(e.tableau.loc["age", "coef"])


def test_cox_retrouve_vrais_parametres(df_grand):
    res = sv.modele_cox(df_grand, "duree_emploi_mois", "evenement_emploi",
                        ["sexe", "qpv", "age", "niveau_formation", "permis_b"],
                        reference={"sexe": "Femme", "niveau_formation": "Niveau 3 (CAP-BEP)"})
    for terme, val in PARAMETRES_VRAIS["cox"].items():
        est, se = res.tableau.loc[terme, ["coef", "erreur_std"]]
        assert abs(est - val) < 4 * se, terme
    assert 0.6 < res.indicateurs["concordance"] < 0.7
    assert isinstance(res.resume(), str)


def test_concordance_force_brute(rng):
    n = 120
    T = np.round(rng.exponential(1, n), 1)
    E = rng.integers(0, 2, n)
    r = np.round(rng.normal(size=n), 1)
    conc = ties = tot = 0
    for i in range(n):
        if E[i] != 1:
            continue
        for j in range(n):
            if T[j] > T[i] or (T[j] == T[i] and E[j] == 0):
                tot += 1
                conc += r[i] > r[j]
                ties += r[i] == r[j]
    assert sv.concordance(T, r, E) == pytest.approx((conc + 0.5 * ties) / tot)


def test_proportionnalite(rng):
    n = 3000
    x = rng.integers(0, 2, n)
    # effet de x qui s'inverse dans le temps : risques non proportionnels
    t = np.where(x == 1, rng.weibull(0.6, n), rng.weibull(2.0, n))
    df = pd.DataFrame({"t": t, "e": 1, "x": x})
    res = sv.modele_cox(df, "t", "e", ["x"])
    assert sv.test_proportionnalite(res).set_index("terme").loc["x", "p_valeur"] < 0.001
    t2 = rng.exponential(1 / np.exp(0.5 * x))
    res2 = sv.modele_cox(df.assign(t=t2), "t", "e", ["x"])
    assert sv.test_proportionnalite(res2).set_index("terme").loc["x", "p_valeur"] > 0.001


def test_cox_strate_et_prediction(df):
    res = sv.modele_cox(df, "duree_emploi_mois", "evenement_emploi", ["qpv", "age"], strate="dispositif")
    assert len(res.risque_cumule_base()) == 3
    p = res.predire_survie(df.head(10), [3, 6, 12], strate="dispositif")
    assert ((p.diff(axis=1).iloc[:, 1:] <= 1e-12).all().all())
    assert ((p >= 0) & (p <= 1)).all().all()
    with pytest.raises(ValueError):
        res.predire_survie(df.head(2), [3])


def test_prediction_sans_covariable_egale_km_approx(petites):
    # modèle de Cox avec covariable nulle : survie de base ~ Kaplan-Meier (Nelson-Aalen)
    d = petites
    res = sv.modele_cox(d, "duree", "evt", ["x"])
    s = res.predire_survie(d.assign(x=res.moyennes[0]).head(1), [0.5, 1.0])
    km = sv.kaplan_meier(d, "duree", "evt").survie_a([0.5, 1.0])["survie"].to_numpy()
    assert np.all(np.abs(s.to_numpy()[0] - km) < 0.1)


def test_incidence_cumulee_une_cause_egale_1_moins_km(petites):
    d = petites.assign(issue=np.where(petites.evt == 1, "evt", "cens"))
    cif = sv.incidence_cumulee(d, "duree", "issue", censure="cens")
    km = sv.kaplan_meier(d, "duree", "evt").tables["Ensemble"]
    fusion = cif.merge(km, on="temps")
    np.testing.assert_allclose(fusion["incidence_cumulee"], 1 - fusion["survie"], atol=1e-12)


def test_incidence_concurrente_somme(rng):
    n = 500
    d = pd.DataFrame({"t": rng.exponential(1, n), "issue": rng.choice(["A", "B", 0], n, p=[.4, .4, .2])})
    cif = sv.incidence_cumulee(d, "t", "issue")
    fin = cif.groupby("issue")["incidence_cumulee"].last().sum()
    assert fin <= 1 + 1e-12


def test_preparer_donnees_survie():
    d = pd.DataFrame({"debut": ["01/01/2025", "01/01/2025", "01/03/2025"],
                      "evt": ["31/01/2025", None, "01/02/2026"]})
    out = sv.preparer_donnees_survie(d, "debut", "evt", "2025-12-31", unite="jours")
    assert out["duree"].tolist() == [30, 364, 305]
    assert out["evenement"].tolist() == [1, 0, 0]


def test_km_par_groupe_et_survie_a(df):
    km = sv.kaplan_meier(df, "duree_emploi_mois", "evenement_emploi", groupe="qpv")
    assert km.logrank["p_valeur"] < 0.05
    sa = km.survie_a([6, 12])
    assert len(sa) == 4 and (sa["ic_bas"] <= sa["survie"]).all() and (sa["survie"] <= sa["ic_haut"]).all()
    assert km.effectifs["n"].sum() == len(df)
