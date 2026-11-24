from obskit import donnees_exemple as de


def test_generation_reproductible():
    a = de.generer_jeunes(300, graine=5)
    b = de.generer_jeunes(300, graine=5)
    assert a.equals(b)
    assert set(a["departement"]) <= set(de.DEPARTEMENTS_OCCITANIE)
    assert a["evenement_emploi"].isin([0, 1]).all()
    assert (a["duree_emploi_mois"] >= 0).all()


def test_manquants():
    d = de.generer_jeunes(1000, graine=1, taux_manquants=0.1)
    assert 0.05 < d["niveau_formation"].isna().mean() < 0.15
