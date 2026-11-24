Modélisation prédictive
=======================

Module : :mod:`obskit.ml` (nécessite ``pip install -e ".[ml]"``).

Objectif : **prédire** au mieux (repérage précoce de jeunes à risque de
décrochage, prévision de volumes), là où les régressions de
:mod:`obskit.regression` visent à **expliquer**.

.. code-block:: python

   from obskit import ml

   mod = ml.modele_predictif(df, cible="acces_emploi",
                             explicatives=["sexe", "age", "qpv", "niveau_formation", "permis_b", "dispositif"],
                             algorithme="gradient_boosting")
   mod.metriques_test           # AUC, exactitude équilibrée, matrice de confusion
   mod.validation_croisee       # stabilité des performances
   mod.importance               # importance par permutation
   mod.predire_proba(nouveaux)

Algorithmes : ``"gradient_boosting"`` (souvent le plus performant, gère les
manquants), ``"foret_aleatoire"``, ``"lineaire"`` (logistique / linéaire),
``"arbre"`` (profondeur 4 : règles lisibles).

Segmentation
------------

.. code-block:: python

   seg = ml.segmentation_kmeans(df, ["age", "score_insertion", "nb_entretiens", "qpv"])
   seg["k"], seg["profils"]

.. warning::
   Exclure des prédicteurs les identifiants, les dates brutes et toute
   information postérieure à l'événement à prédire (fuite d'information).
   Un outil de repérage individuel doit faire l'objet d'une réflexion éthique
   et RGPD avant tout usage opérationnel.
