import numpy as np
import pandas as pd
import pytest

from obskit._encodage import Encodeur, construire_matrice


def _df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0],
                         "s": ["F", "H", "H", "F", "H"],
                         "n": pd.Categorical(["a", "b", "c", "a", "b"], categories=["c", "b", "a"])})


def test_colonnes_et_reference_frequente():
    X, enc = construire_matrice(_df(), ["x", "s", "n"])
    assert enc.references_["s"] == "H"
    assert list(X.columns)[:3] == ["const", "x", "s[F]"]
    assert X["s[F]"].tolist() == [1, 0, 0, 1, 0]


def test_reference_premiere_respecte_ordre_categories():
    X, enc = construire_matrice(_df(), ["n"], reference_defaut="premiere")
    assert enc.references_["n"] == "c"
    assert list(X.columns) == ["const", "n[b]", "n[a]"]


def test_reference_explicite_et_erreur():
    X, enc = construire_matrice(_df(), ["s"], reference={"s": "F"})
    assert "s[H]" in X.columns
    with pytest.raises(ValueError):
        construire_matrice(_df(), ["s"], reference={"s": "Z"})


def test_interactions():
    X, enc = construire_matrice(_df(), ["x", "s"], interactions=[("x", "s")])
    assert "x:s[F]" in X.columns
    np.testing.assert_allclose(X["x:s[F]"], X["x"] * X["s[F]"])


def test_modalite_unique_refusee():
    d = pd.DataFrame({"s": ["a"] * 4, "x": range(4)})
    with pytest.raises(ValueError):
        Encodeur(["s"]).fit(d)


def test_modalite_inconnue_avertit():
    enc = Encodeur(["s"]).fit(_df())
    with pytest.warns(UserWarning):
        X = enc.transform(pd.DataFrame({"s": ["F", "Z"]}))
    assert X["s[F]"].tolist() == [1.0, 0.0]


def test_categorielle_forcee():
    d = pd.DataFrame({"annee": [2023, 2024, 2025, 2024]})
    X = Encodeur(["annee"], categorielles=["annee"], reference={"annee": 2023}).fit_transform(d)
    assert list(X.columns) == ["const", "annee[2024]", "annee[2025]"]
