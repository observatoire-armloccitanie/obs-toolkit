import pandas as pd

from obskit import correlation, descriptif, factoriel, regression, survie
from obskit import visualisation as vz


def test_style_et_barres(df, tmp_path):
    vz.appliquer_style()
    tf = descriptif.tableau_frequences(df, "niveau_formation", total=False)
    fig, ax = vz.diagramme_barres(tf["pct"], format_etiquette="{:.1f} %", titre="t", source="s")
    assert len(ax.patches) == 4
    tc = descriptif.tableau_croise(df, "niveau_formation", "sexe", marges=False)
    fig, ax = vz.diagramme_barres(tc, horizontal=False)
    assert len(ax.patches) == 8
    sorties = vz.enregistrer(fig, tmp_path / "barres", formats=("png", "svg"))
    assert all(pd.io.common.file_exists(s) for s in sorties)


def test_graphiques_divers(df):
    km = survie.kaplan_meier(df, "duree_emploi_mois", "evenement_emploi", groupe="qpv")
    km.tracer(incidence=True, effectifs_a_risque=[0, 6, 12])
    cox = survie.modele_cox(df, "duree_emploi_mois", "evenement_emploi", ["qpv", "age"])
    vz.forest_plot(cox)
    lin = regression.regression_lineaire(df, "score_insertion", ["qpv", "age"])
    vz.forest_plot(lin, exposant=False)
    vz.heatmap_correlation(correlation.matrice_correlation(df, ["age", "score_insertion", "nb_entretiens"])["r"])
    log = regression.regression_logistique(df, "acces_emploi", ["qpv", "age"])
    vz.courbe_roc(log.y, log.valeurs_ajustees, label="modèle")
    vz.evolution_temporelle(descriptif.evolution(df, "date_entree", "M"))
    vz.histogramme(df, "score_insertion", groupe="sexe")
    vz.boites(df, "score_insertion", "dispositif")
    acm = factoriel.acm(df, ["sexe", "qpv", "dispositif"], variables_sup=["situation_sortie"])
    acm.tracer()
    acm.tracer(quoi="individus")
    factoriel.acp(df, ["age", "score_insertion", "nb_entretiens"]).tracer()
    factoriel.afc(pd.crosstab(df.dispositif, df.situation_sortie)).tracer()
