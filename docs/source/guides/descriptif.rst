Statistiques descriptives et tests
==================================

Modules : :mod:`obskit.descriptif`, :mod:`obskit.inference`.

Tri à plat
----------

.. code-block:: python

   from obskit import descriptif as ds, inference as inf, donnees_exemple
   df = donnees_exemple.generer_jeunes(3000)

   ds.tableau_frequences(df, "niveau_formation", tri="modalite")
   ds.tableau_frequences(df, "niveau_formation", poids="poids", inclure_manquants=True)

Le tableau contient effectifs, pourcentages, pourcentages cumulés et
**intervalle de confiance** du pourcentage (méthode de Wilson, fiable même pour
de petits effectifs).

Tri croisé
----------

.. code-block:: python

   tc = ds.tableau_croise(df, "niveau_formation", "situation_sortie", pourcentage="ligne")
   tc.attrs["effectifs"]     # effectifs bruts avec marges
   tc.attrs["test"]          # {'test': 'chi2', 'p_valeur': ..., 'v_cramer': ...}

Lecture du V de Cramér : < 0,1 lien faible, 0,1–0,3 modéré, > 0,3 fort
(ordres de grandeur indicatifs). Pour savoir **quelles cases** portent l'écart
à l'indépendance, regarder les résidus standardisés :

.. code-block:: python

   inf.test_independance(df, "niveau_formation", "situation_sortie")["residus_standardises"]

Un résidu > 2 (ou < −2) signale une sur- (sous-) représentation.

« Tableau 1 »
-------------

.. code-block:: python

   ds.tableau_un(df, groupe="dispositif",
                 variables=["sexe", "age", "qpv", "niveau_formation", "score_insertion"],
                 continues="mediane", correction="holm")

Une colonne par groupe, une colonne « Ensemble », une p-valeur par variable
(χ² pour les qualitatives, test de comparaison adapté pour les quantitatives).

Statistiques quantitatives
--------------------------

.. code-block:: python

   ds.statistiques_descriptives(df, ["age", "score_insertion"], par="departement", poids="poids")

Évolutions
----------

.. code-block:: python

   ev = ds.evolution(df, date="date_entree", frequence="M")      # entrées mensuelles
   ev[["valeur", "variation_pct", "glissement_annuel_pct"]]
   ds.evolution(df, "date_entree", "Q", par="dispositif")        # une colonne par dispositif

Comparer des groupes
--------------------

.. code-block:: python

   inf.comparer_groupes(df, "score_insertion", "sexe")          # Welch ou Mann-Whitney
   inf.comparer_groupes(df, "score_insertion", "dispositif")    # ANOVA de Welch ou Kruskal
   inf.comparer_proportions(k1=420, n1=1000, k2=350, n2=1000)   # taux d'accès QPV / hors QPV

Chaque résultat inclut une **taille d'effet** (d de Cohen, r rang-bisérial,
η², ε²). Pour plusieurs tests simultanés : :func:`~obskit.inference.corriger_pvaleurs`.

Préparer une enquête
--------------------

.. code-block:: python

   inf.taille_echantillon_proportion(marge=0.03)                    # ± 3 points -> 1068 répondants
   inf.taille_echantillon_proportion(marge=0.05, population=2500)   # population finie
