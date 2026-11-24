Démarrage rapide
================

Ce parcours utilise des **données simulées** (:func:`obskit.donnees_exemple.generer_jeunes`)
qui imitent une extraction i-milo : on peut donc copier-coller chaque bloc.

.. code-block:: python

   import obskit as ok

   df = ok.donnees_exemple.generer_jeunes(3000)
   ok.visualisation.appliquer_style()

1. Explorer
-----------

.. code-block:: python

   ok.descriptif.resume_variables(df)            # types, manquants, modalités
   ok.preparation.detecter_types(df)             # type statistique proposé
   ok.descriptif.tableau_frequences(df, "niveau_formation", tri="modalite")

2. Croiser et tester
--------------------

.. code-block:: python

   tc = ok.descriptif.tableau_croise(df, "niveau_formation", "situation_sortie")
   tc.attrs["test"]          # χ², p-valeur, V de Cramér

   ok.descriptif.tableau_un(df, groupe="dispositif",
                            variables=["sexe", "age", "qpv", "niveau_formation"])

3. Repérer les déterminants d'une cible
---------------------------------------

.. code-block:: python

   ok.correlation.associations_avec_cible(df, cible="acces_emploi",
       explicatives=["sexe", "age", "qpv", "niveau_formation", "permis_b", "departement"])

4. Modéliser « toutes choses égales par ailleurs »
----------------------------------------------------

.. code-block:: python

   res = ok.regression.regression_logistique(
       df, cible="acces_emploi",
       explicatives=["sexe", "age", "qpv", "niveau_formation", "permis_b"],
       reference={"niveau_formation": "Niveau 3 (CAP-BEP)"},
   )
   print(res.resume())
   res.tableau_complet()                         # format publication avec « Réf. »
   ok.regression.effets_marginaux_moyens(res)    # en points de probabilité
   ok.visualisation.forest_plot(res)

5. Durées : quand l'accès à l'emploi a-t-il lieu ?
--------------------------------------------------

.. code-block:: python

   km = ok.survie.kaplan_meier(df, "duree_emploi_mois", "evenement_emploi", groupe="qpv")
   km.medianes
   km.tracer(incidence=True)

   cox = ok.survie.modele_cox(df, "duree_emploi_mois", "evenement_emploi",
                              ["sexe", "age", "qpv", "niveau_formation", "permis_b"])
   print(cox.resume())
   ok.survie.test_proportionnalite(cox)

6. Exporter
-----------

.. code-block:: python

   ok.io.exporter_excel(
       {"Fréquences": ok.descriptif.tableau_frequences(df, "niveau_formation"),
        "Modèle logistique": res.tableau_complet(),
        "Cox": cox.tableau},
       "sorties/resultats.xlsx",
       notes={"Fréquences": "Source : i-milo, jeunes accompagnés 2023-2025, traitement Observatoire ARML."},
   )

Un script complet est disponible dans ``examples/analyse_complete.py``.
