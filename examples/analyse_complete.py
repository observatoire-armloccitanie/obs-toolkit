"""Exemple d'analyse complète avec obskit, sur données simulées.

Exécution :  python examples/analyse_complete.py
Sorties   :  dossier ``sorties/`` (classeur Excel + graphiques PNG).

Pour l'adapter à une vraie extraction, remplacer la génération des données par
``ok.io.lire_fichier("mon_export.csv", nettoyer_noms=True)`` et ajuster les noms
de colonnes dans le dictionnaire CONFIG.
"""

from pathlib import Path

import obskit as ok

CONFIG = {
    "cible_binaire": "acces_emploi",
    "duree": "duree_emploi_mois",
    "evenement": "evenement_emploi",
    "explicatives": ["sexe", "age", "qpv", "niveau_formation", "permis_b", "dispositif"],
    "references": {"niveau_formation": "Niveau 3 (CAP-BEP)", "dispositif": "PACEA"},
    "groupe": "dispositif",
    "source": "Source : données simulées - traitement Observatoire ARML Occitanie",
}
SORTIES = Path("sorties")


def main() -> None:
    SORTIES.mkdir(exist_ok=True)
    ok.visualisation.appliquer_style()
    vz = ok.visualisation
    c = CONFIG

    # 1. Données
    df = ok.donnees_exemple.generer_jeunes(5000, graine=1)
    df["tranche_age"] = ok.preparation.tranches_age(df["age"])

    # 2. Descriptif
    tableaux = {
        "Variables": ok.descriptif.resume_variables(df),
        "Niveau formation": ok.descriptif.tableau_frequences(df, "niveau_formation", tri="modalite"),
        "Sortie x formation": ok.descriptif.tableau_croise(df, "niveau_formation", "situation_sortie"),
        "Tableau 1": ok.descriptif.tableau_un(df, c["groupe"], ["sexe", "tranche_age", "qpv", "niveau_formation"]),
        "Entrées mensuelles": ok.descriptif.evolution(df, "date_entree", "M"),
        "Liens avec la cible": ok.correlation.associations_avec_cible(df, c["cible_binaire"], c["explicatives"]),
    }
    tf = ok.descriptif.tableau_frequences(df, "niveau_formation", tri="modalite", total=False)
    fig, _ = vz.diagramme_barres(tf["pct"], trier=False, format_etiquette="{:.1f} %",
                                 titre="Niveau de formation à l'entrée", source=c["source"])
    vz.enregistrer(fig, SORTIES / "niveau_formation")

    # 3. Régression logistique
    log = ok.regression.regression_logistique(df, c["cible_binaire"], c["explicatives"], reference=c["references"])
    print(log.resume())
    tableaux["Logistique"] = log.tableau_complet()
    tableaux["Effets marginaux"] = ok.regression.effets_marginaux_moyens(log)
    fig, _ = vz.forest_plot(log, titre="Accès à l'emploi : odds ratios")
    vz.enregistrer(fig, SORTIES / "forest_logistique")

    # 4. Survie
    km = ok.survie.kaplan_meier(df, c["duree"], c["evenement"], groupe="niveau_formation")
    tableaux["KM médianes"] = km.medianes
    fig, _ = km.tracer(incidence=True, xlabel="Mois depuis l'entrée", titre="Accès à l'emploi")
    vz.enregistrer(fig, SORTIES / "kaplan_meier")
    cox = ok.survie.modele_cox(df, c["duree"], c["evenement"],
                               [v for v in c["explicatives"] if v != "dispositif"],
                               reference=c["references"], strate="dispositif")
    print(cox.resume())
    tableaux["Cox"] = cox.tableau
    tableaux["Proportionnalité"] = ok.survie.test_proportionnalite(cox)

    # 5. Typologie (ACM + CAH)
    acm = ok.factoriel.acm(df, ["sexe", "tranche_age", "qpv", "niveau_formation", "permis_b"],
                           variables_sup=["situation_sortie"], correction_benzecri=True)
    cah = ok.factoriel.classification(acm.individus["coord"].iloc[:, :4])
    tableaux["ACM valeurs propres"] = acm.valeurs_propres
    tableaux["Classes"] = ok.factoriel.caracteriser_classes(
        df, cah["classes"], ["sexe", "tranche_age", "qpv", "niveau_formation", "permis_b", "situation_sortie"])
    fig, _ = acm.tracer()
    vz.enregistrer(fig, SORTIES / "acm")

    # 6. Export
    chemin = ok.io.exporter_excel(tableaux, SORTIES / "resultats.xlsx", notes={k: c["source"] for k in tableaux})
    print(f"Résultats exportés dans {chemin}")


if __name__ == "__main__":
    main()
