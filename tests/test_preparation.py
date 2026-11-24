import numpy as np
import pandas as pd
import pytest

from obskit import preparation as pr


def test_nettoyer_noms():
    d = pd.DataFrame(columns=["Date d'entrée en CEJ", "Âge", "âge", " N° jeune "])
    assert list(pr.nettoyer_noms_colonnes(d).columns) == ["date_d_entree_en_cej", "age", "age_2", "n_jeune"]


def test_tranches_age():
    t = pr.tranches_age(pd.Series([16, 17, 18, 21, 22, 25, 26, 29]))
    assert t.tolist() == ["16-17 ans", "16-17 ans", "18-21 ans", "18-21 ans", "22-25 ans", "22-25 ans",
                          "26 ans et +", "26 ans et +"]
    assert t.cat.ordered
    t2 = pr.tranches_age([15, 20], bornes=(16, 26))
    assert t2.tolist() == ["moins de 16 ans", "16-25 ans"]


def test_calculer_age():
    a = pr.calculer_age(pd.Series(["15/06/2005", "16/06/2005"]), "2023-06-15")
    assert a.tolist() == [18, 17]


def test_duree_entre():
    d = pr.duree_entre(pd.Series(["01/01/2024"]), pd.Series(["31/01/2024"]), "jours")
    assert d.iloc[0] == 30
    assert pr.duree_entre(["2024-01-01"], "2025-01-01", "annees").iloc[0] == pytest.approx(366 / 365.25)


def test_recoder():
    s = pd.Series(["CDI", "CDD > 6 mois", "Intérim", "Formation", None])
    r = pr.recoder(s, {("CDI", "CDD > 6 mois"): "Emploi durable", "Intérim": "Emploi précaire"}, autre="Autre")
    assert r.tolist()[:4] == ["Emploi durable", "Emploi durable", "Emploi précaire", "Autre"]
    assert pd.isna(r.iloc[4])
    o = pr.recoder(s.dropna(), {"CDI": "E"}, ordre=["E", "CDD > 6 mois", "Intérim", "Formation"])
    assert o.cat.ordered


def test_regrouper_rares():
    s = pd.Series(["a"] * 50 + ["b"] * 45 + ["c"] * 3 + ["d"] * 2)
    r = pr.regrouper_modalites_rares(s, seuil=0.05)
    assert set(r) == {"a", "b", "Autre"} and (r == "Autre").sum() == 5


def test_convertir_binaire():
    s = pd.Series(["Oui", "non", "O", None, "N"])
    assert pr.convertir_binaire(s).tolist()[:3] == [1, 0, 1]
    with pytest.raises(ValueError):
        pr.convertir_binaire(pd.Series(["peut-être"]))
    assert pr.convertir_binaire(pd.Series(["Emploi", "Autre"]), "Emploi").tolist() == [1, 0]


def test_detecter_types(df):
    t = pr.detecter_types(df).set_index("variable")["type_detecte"]
    assert t["id_jeune"] == "identifiant"
    assert t["qpv"] == "binaire"
    assert t["niveau_formation"] == "categorielle"
    assert t["score_insertion"] == "numerique"
    assert t["date_entree"] == "date"


def test_imputer():
    d = pd.DataFrame({"x": [1.0, np.nan, 3.0], "c": ["a", None, "a"],
                      "k": pd.Categorical(["u", None, "v"])})
    out = pr.imputer(d)
    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    assert out["c"].iloc[1] == "Non renseigné" and out["k"].iloc[1] == "Non renseigné"
    out2 = pr.imputer(d, {"c": "mode", "x": 0})
    assert out2["c"].iloc[1] == "a" and out2["x"].iloc[1] == 0


def test_discretiser_et_ordonner(rng):
    q = pr.discretiser(rng.normal(size=100), 4)
    assert q.value_counts().tolist() == [25, 25, 25, 25]
    o = pr.ordonner_modalites(pd.Series(["b", "a"]), ["a", "b"])
    assert list(o.cat.categories) == ["a", "b"]
    with pytest.raises(ValueError):
        pr.ordonner_modalites(pd.Series(["z"]), ["a"])


def test_convertir_dates():
    d = pr.convertir_dates(pd.DataFrame({"d": ["03/02/2025", "xx"]}), ["d"])
    assert d["d"].iloc[0] == pd.Timestamp("2025-02-03") and pd.isna(d["d"].iloc[1])
