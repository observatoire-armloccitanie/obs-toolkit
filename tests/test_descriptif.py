import numpy as np
import pandas as pd
import pytest

from obskit import descriptif as ds


def test_resume_et_manquants():
    from obskit.donnees_exemple import generer_jeunes

    d = generer_jeunes(500, graine=1, taux_manquants=0.1)
    r = ds.resume_variables(d)
    assert len(r) == d.shape[1]
    vm = ds.valeurs_manquantes(d)
    assert set(vm.index) == {"niveau_formation", "permis_b", "date_acces_emploi"}


def test_tableau_frequences(df):
    tf = ds.tableau_frequences(df, "niveau_formation", tri="modalite")
    assert list(tf.index[:-1]) == list(df.niveau_formation.cat.categories)
    assert tf.loc["Total", "effectif"] == len(df)
    assert tf["pct"].iloc[:-1].sum() == pytest.approx(100)
    assert (tf["ic_bas"].iloc[:-1] < tf["pct"].iloc[:-1]).all()
    tp = ds.tableau_frequences(df, "sexe", poids="poids", total=False)
    attendu = df.groupby("sexe")["poids"].sum() / df["poids"].sum() * 100
    assert tp.loc["Femme", "pct"] == pytest.approx(attendu["Femme"])


def test_tableau_frequences_manquants():
    d = pd.DataFrame({"x": ["a", "b", None, "a"]})
    tf = ds.tableau_frequences(d, "x", inclure_manquants=True, ic=False)
    assert tf.loc["Non renseigné", "effectif"] == 1


def test_tableau_croise(df):
    tc = ds.tableau_croise(df, "niveau_formation", "situation_sortie", decimales=None)
    np.testing.assert_allclose(tc.sum(axis=1).to_numpy(), 200)
    assert tc.attrs["test"]["p_valeur"] < 0.05
    tcc = ds.tableau_croise(df, "sexe", "qpv", pourcentage="colonne", marges=False, decimales=None)
    np.testing.assert_allclose(tcc.sum(axis=0), 100)
    eff = ds.tableau_croise(df, "sexe", "qpv", pourcentage=None)
    assert eff.loc["Ensemble", "Ensemble"] == len(df)


def test_statistiques_descriptives(df):
    s = ds.statistiques_descriptives(df, ["age", "score_insertion"])
    assert s.loc["age", "moyenne"] == pytest.approx(df.age.mean())
    assert s.loc["age", "mediane"] == pytest.approx(df.age.median())
    sp = ds.statistiques_descriptives(df, ["age"], par="sexe", poids="poids")
    f = df[df.sexe == "Femme"]
    assert sp.loc[("Femme", "age"), "moyenne"] == pytest.approx(np.average(f.age, weights=f.poids))


def test_quantiles_ponderes(rng):
    x = rng.normal(size=1001)
    assert ds.quantiles_ponderes(x, np.ones(1001), [0.5])[0] == pytest.approx(np.median(x), abs=1e-2)


def test_tableau_un(df):
    t1 = ds.tableau_un(df, "dispositif", ["sexe", "age", "niveau_formation", "score_insertion"])
    assert {"CEJ", "PACEA", "Autre", "Ensemble", "p_valeur"} <= set(t1.columns)
    assert ("niveau_formation", "Infra 3") in t1.index
    t0 = ds.tableau_un(df, None, ["sexe", "age"], continues="mediane")
    assert list(t0.columns) == ["Ensemble"]


def test_evolution(df):
    ev = ds.evolution(df, "date_entree", "M")
    assert ev["valeur"].sum() == len(df)
    assert len(ev) == 36
    assert "glissement_annuel_pct" in ev.columns
    evp = ds.evolution(df, "date_entree", "Y", par="dispositif")
    assert evp.to_numpy().sum() == len(df)
    cum = ds.evolution(df, "date_entree", "Q", cumul=True)
    assert cum["valeur"].iloc[-1] == len(df)
