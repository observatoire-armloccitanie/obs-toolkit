import pandas as pd
import pytest

from obskit import io


def test_csv_obiee_like(tmp_path, df):
    p = tmp_path / "export.csv"
    df.head(30).rename(columns={"age": "Âge du jeune"}).to_csv(p, sep=";", encoding="cp1252", decimal=",", index=False)
    lu = io.lire_fichier(p, nettoyer_noms=True, dtype={"departement": str})
    assert "age_du_jeune" in lu.columns
    assert lu["score_insertion"].dtype.kind == "f"
    assert lu["departement"].str.len().eq(2).all()
    pd.testing.assert_series_equal(lu["score_insertion"], df.head(30)["score_insertion"], check_names=False)


def test_csv_virgule_utf8(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"é": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    assert list(io.lire_fichier(p).columns) == ["é", "b"]


def test_excel_aller_retour(tmp_path, df):
    from obskit import descriptif

    chemin = io.exporter_excel(
        {"Fréquences": descriptif.tableau_frequences(df, "sexe"),
         "Tableau 1": descriptif.tableau_un(df, "sexe", ["age", "qpv"]),
         "Fréquences ": df.head(3)},
        tmp_path / "sortie" / "r.xlsx", notes={"Fréquences": "Source : données simulées"})
    xl = pd.ExcelFile(chemin)
    assert len(xl.sheet_names) == 3
    relu = pd.read_excel(chemin, sheet_name=xl.sheet_names[0], index_col=0)
    assert "effectif" in relu.columns
    assert io.lire_fichier(chemin).shape[0] > 0


def test_erreurs(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.lire_fichier(tmp_path / "absent.csv")
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError):
        io.lire_fichier(tmp_path / "x.json")
