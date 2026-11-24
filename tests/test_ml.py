import pytest

pytest.importorskip("sklearn")

from obskit import ml  # noqa: E402

X = ["sexe", "qpv", "age", "niveau_formation", "permis_b", "dispositif"]


@pytest.mark.parametrize("algo", ["gradient_boosting", "lineaire", "arbre"])
def test_classification(df, algo):
    r = ml.modele_predictif(df, "acces_emploi", X, algorithme=algo, cv=3, importance=(algo == "lineaire"))
    assert r.tache == "classification"
    assert r.metriques_test["auc"] > 0.6
    assert len(r.validation_croisee) == 3
    p = r.predire_proba(df.head(5))
    assert p.shape == (5, 2)


def test_regression_et_importance(df):
    r = ml.modele_predictif(df, "score_insertion", X, algorithme="lineaire", cv=3)
    assert r.tache == "regression" and r.metriques_test["r2"] > 0.25
    assert r.importance["variable"].iloc[0] in {"niveau_formation", "permis_b", "age"}
    assert len(r.predire(df.head(3))) == 3


def test_multiclasse(df):
    r = ml.modele_predictif(df, "situation_sortie", X, algorithme="foret_aleatoire", cv=2, importance=False)
    assert len(r.classes) == 4


def test_segmentation(df):
    s = ml.segmentation_kmeans(df, ["age", "score_insertion", "qpv"], k_max=4)
    assert 2 <= s["k"] <= 4
    assert s["profils"]["effectif"].sum() == len(df)
    s3 = ml.segmentation_kmeans(df, ["age", "score_insertion"], k=3)
    assert s3["classes"].nunique() == 3
