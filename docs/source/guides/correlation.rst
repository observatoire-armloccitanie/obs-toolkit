Corrélations et associations
============================

Module : :mod:`obskit.correlation`.

Variables quantitatives
-----------------------

.. code-block:: python

   from obskit import correlation as cr, visualisation as vz
   m = cr.matrice_correlation(df, ["age", "score_insertion", "nb_entretiens"], methode="spearman")
   m["r"], m["p_valeur"], m["n"]
   vz.heatmap_correlation(m["r"])

   cr.matrice_correlation(df, methode="pearson", format_long=True)   # paires triées par |r|

Les p-valeurs sont corrigées par défaut (Holm). **Spearman** est préférable en
présence de valeurs extrêmes ou de relations monotones non linéaires.

Variables de tous types
-----------------------

:func:`~obskit.correlation.matrice_association` combine V de Cramér
(qualitative x qualitative), rapport de corrélation η (qualitative x
quantitative) et \|r\| : utile pour repérer des variables redondantes avant
une régression.

.. code-block:: python

   vz.heatmap_correlation(cr.matrice_association(df, ["sexe", "qpv", "age", "niveau_formation", "score_insertion"]))

Déterminants d'une cible
------------------------

.. code-block:: python

   cr.associations_avec_cible(df, cible="acces_emploi")          # toutes les colonnes
   cr.associations_avec_cible(df, cible="situation_sortie",
                              explicatives=["sexe", "qpv", "age", "niveau_formation"])

Contrôler un facteur de confusion
---------------------------------

.. code-block:: python

   cr.correlation_partielle(df, "nb_entretiens", "score_insertion", controles=["duree_accompagnement_mois"])

.. note::
   Une corrélation, même partielle, ne mesure pas un effet causal : voir
   :doc:`causalite`.
