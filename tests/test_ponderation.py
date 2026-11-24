import numpy as np
import pytest

from obskit import ponderation as po

MARGES = {"sexe": {"Femme": 0.5, "Homme": 0.5}, "qpv": {True: 0.3, False: 0.7},
          "dispositif": {"CEJ": 50, "PACEA": 30, "Autre": 20}}


def test_raking_reproduit_marges(df):
    w = po.redressement_raking(df, MARGES)
    assert w.sum() == pytest.approx(len(df))
    cmp = po.comparer_marges(df.assign(w=w), MARGES, "w")
    assert cmp["ecart_points"].abs().max() < 1e-6
    w2 = po.redressement_raking(df, MARGES, total=100000, poids_initiaux="poids")
    assert w2.sum() == pytest.approx(100000)


def test_raking_borne(df):
    w = po.redressement_raking(df, {"qpv": {True: 0.6, False: 0.4}}, bornes=(0.5, 2.0))
    assert w.min() >= 0.5 - 1e-9 and w.max() <= 2.0 + 1e-9


def test_raking_modalite_manquante(df):
    with pytest.raises(ValueError):
        po.redressement_raking(df, {"sexe": {"Femme": 1.0}})


def test_effet_plan():
    assert po.effet_plan(np.ones(10))["deff_kish"] == pytest.approx(1)
    e = po.effet_plan([1, 1, 2, 2])
    assert e["deff_kish"] == pytest.approx(4 * 10 / 36)
