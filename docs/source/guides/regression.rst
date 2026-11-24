Modèles de régression
=====================

Module : :mod:`obskit.regression`.

Principes communs
-----------------

Toutes les fonctions partagent la même signature :

.. code-block:: python

   regression_xxx(df, cible, explicatives=None, reference=None, interactions=None,
                  poids=None, erreurs="classique", grappes=None, alpha=0.05)

* ``explicatives=None`` utilise toutes les autres colonnes ;
* les variables texte, catégorielles et booléennes sont codées en indicatrices ;
  la **modalité de référence** est par défaut la plus fréquente et se fixe par
  ``reference={"niveau_formation": "Niveau 3 (CAP-BEP)"}`` ;
* une variable numérique codée (année, code) se traite en catégorielle avec
  ``categorielles=["annee"]`` ;
* les lignes incomplètes sont exclues (avertissement + ``indicateurs["n_exclus"]``) ;
* ``erreurs="HC3"`` : erreurs-types robustes à l'hétéroscédasticité ;
  ``grappes="mission_locale"`` : erreurs robustes par grappe (jeunes d'une même
  structure non indépendants).

Le résultat (:class:`~obskit.regression.ResultatRegression`) expose
``tableau``, ``indicateurs``, ``resume()``, ``tableau_complet()``,
``predire()``, ``residus``…

Régression logistique
---------------------

.. code-block:: python

   from obskit import regression as rg

   res = rg.regression_logistique(
       df, cible="acces_emploi",
       explicatives=["sexe", "age", "qpv", "niveau_formation", "permis_b", "dispositif"],
       reference={"niveau_formation": "Niveau 3 (CAP-BEP)", "dispositif": "PACEA"},
   )
   print(res.resume())
   res.tableau[["odds_ratio", "odds_ratio_ic_bas", "odds_ratio_ic_haut", "p_valeur"]]

Cible non codée 0/1 : ``rg.regression_logistique(df, "situation_sortie", ..., modalite_positive="Emploi")``.

**Effets marginaux moyens** - plus lisibles que les odds ratios et comparables
entre modèles : « avoir le permis B augmente en moyenne la probabilité d'accès
à l'emploi de 13 points » (exemple fictif).

.. code-block:: python

   rg.effets_marginaux_moyens(res)

**Qualité et diagnostic**

.. code-block:: python

   rg.test_global_variables(res)          # chaque variable (toutes modalités) est-elle utile ?
   rg.performance_classification(res)     # AUC, matrice de confusion, sensibilité…
   rg.test_hosmer_lemeshow(res)           # adéquation
   rg.vif(df, ["sexe", "age", "qpv", "niveau_formation"])   # colinéarité

Interactions
------------

L'effet du QPV est-il le même pour les femmes et les hommes ?

.. code-block:: python

   res = rg.regression_logistique(df, "acces_emploi", ["sexe", "qpv", "age"],
                                  interactions=[("sexe", "qpv")])

Régression linéaire
-------------------

.. code-block:: python

   lin = rg.regression_lineaire(df, "score_insertion", ["sexe", "age", "qpv", "niveau_formation"],
                                erreurs="HC3")
   lin.indicateurs["r2"], lin.indicateurs["r2_ajuste"]
   lin.residus                         # pour les graphiques de diagnostic

Logistique multinomiale
-----------------------

Pour une cible à plus de deux modalités non ordonnées :

.. code-block:: python

   mn = rg.regression_logistique_multinomiale(df, "situation_sortie",
                                              ["qpv", "permis_b", "niveau_formation"],
                                              reference_cible="Sans solution")
   print(mn.resume())          # RRR = rapports de risques relatifs vs la référence
   mn.predire(df.head())       # probabilités de chaque situation

Poisson (comptages)
-------------------

.. code-block:: python

   po = rg.regression_poisson(df, "nb_entretiens", ["dispositif", "qpv"],
                              exposition="duree_accompagnement_mois", erreurs="HC1")
   po.tableau["irr"]           # rapports de taux (entretiens par mois)

Comparer et sélectionner
------------------------

.. code-block:: python

   m1 = rg.regression_logistique(df, "acces_emploi", ["sexe", "age"])
   m2 = rg.regression_logistique(df, "acces_emploi", ["sexe", "age", "niveau_formation"])
   rg.comparer_modeles({"socio-démo": m1, "+ diplôme": m2})
   rg.test_rapport_vraisemblance(m1, m2)

   modele, etapes = rg.selection_pas_a_pas(df, "acces_emploi", candidates=[...], critere="bic")

.. warning::
   La sélection automatique est un outil exploratoire : les p-valeurs du
   modèle final sont optimistes. Privilégier un modèle construit à partir
   d'hypothèses sociologiques.
