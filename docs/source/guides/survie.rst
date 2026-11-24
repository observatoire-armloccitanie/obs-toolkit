Analyse de survie
=================

Module : :mod:`obskit.survie`.

L'analyse de survie étudie **le temps jusqu'à un événement** (accès à l'emploi,
sortie de dispositif) en tenant compte des jeunes pour qui l'événement n'est pas
encore observé à la date d'extraction (**censure**). Ignorer la censure - par
exemple calculer une durée moyenne sur les seuls jeunes sortis - biaise
fortement les résultats.

Préparer durée et événement
---------------------------

.. code-block:: python

   from obskit import survie as sv

   df = sv.preparer_donnees_survie(
       df, date_debut="date_entree", date_evenement="date_acces_emploi",
       date_fin_observation="2026-06-30",      # ou une colonne (date de fin de suivi individuelle)
       unite="mois",
   )
   # -> colonnes "duree" et "evenement" (1 = événement observé, 0 = censuré)

Kaplan-Meier
------------

.. code-block:: python

   km = sv.kaplan_meier(df, "duree_emploi_mois", "evenement_emploi", groupe="niveau_formation")
   km.medianes                 # durée médiane d'accès à l'emploi et IC par groupe
   km.survie_a([3, 6, 12])     # part n'ayant pas encore accédé à l'emploi à 3, 6, 12 mois
   km.rmst(24)                 # durée moyenne restreinte à 24 mois
   km.logrank                  # test d'égalité des courbes
   km.tracer(incidence=True, effectifs_a_risque=[0, 6, 12, 18, 24])

``incidence=True`` représente ``1 − S(t)``, la part cumulée de jeunes ayant
accédé à l'emploi : plus intuitif pour un événement « positif ».

Modèle de Cox
-------------

.. code-block:: python

   cox = sv.modele_cox(df, "duree_emploi_mois", "evenement_emploi",
                       ["sexe", "age", "qpv", "niveau_formation", "permis_b"],
                       reference={"niveau_formation": "Niveau 3 (CAP-BEP)"})
   print(cox.resume())
   cox.tableau[["hr", "hr_ic_bas", "hr_ic_haut", "p_valeur"]]

Un **rapport de risques (HR)** de 1,6 pour le permis B signifie qu'à tout
instant, le « risque » (ici la chance) d'accéder à l'emploi est 1,6 fois plus
élevé pour les titulaires du permis, à caractéristiques égales.

**Vérifier la proportionnalité des risques**

.. code-block:: python

   sv.test_proportionnalite(cox)

Si une variable ne respecte pas l'hypothèse (p < 0,05), la passer en strate :

.. code-block:: python

   cox = sv.modele_cox(df, "duree_emploi_mois", "evenement_emploi",
                       ["sexe", "age", "qpv"], strate="dispositif")

**Prédire** des courbes de survie individuelles :

.. code-block:: python

   cox.predire_survie(nouveaux_jeunes, temps=[6, 12], strate="dispositif")

Ex aequo : ``methode="efron"`` (défaut, comme R et lifelines) est plus précis
que ``"breslow"`` quand les durées sont arrondies (mois).

Risques concurrents
-------------------

Quand plusieurs issues s'excluent (emploi, formation, abandon), ``1 − KM``
surestime l'incidence de chacune. Utiliser l'estimateur d'Aalen-Johansen :

.. code-block:: python

   cif = sv.incidence_cumulee(df, "duree", "type_sortie", censure="En cours", groupe="qpv")
