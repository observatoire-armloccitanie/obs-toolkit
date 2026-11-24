Évaluer l'effet d'un dispositif
===============================

Module : :mod:`obskit.causalite`.

Problème
--------

Les jeunes qui réalisent une PMSMP ou entrent en CEJ ne sont pas tirés au sort :
ils diffèrent des autres (âge, diplôme, mobilité…). La différence brute de taux
d'accès à l'emploi mélange l'effet du dispositif et ces différences de profil
(**biais de sélection**).

Démarche
--------

1. Choisir les covariables : caractéristiques **mesurées avant** le traitement
   qui influencent à la fois l'entrée dans le dispositif et le résultat.
2. Vérifier le déséquilibre initial, estimer l'effet, vérifier l'équilibre après ajustement.
3. Discuter la robustesse à un facteur non observé.

.. code-block:: python

   from obskit import causalite as ca

   cov = ["sexe", "age", "qpv", "niveau_formation", "permis_b"]
   ca.equilibre_covariables(df, "a_fait_pmsmp", cov)          # SMD avant ajustement

   eff = ca.estimer_effet(df, traitement="a_fait_pmsmp", resultat="acces_emploi",
                          covariables=cov, methode="aipw", estimand="ATE")
   print(eff.resume())
   eff.equilibre_apres                                        # |SMD| < 0,1 attendu

Méthodes disponibles
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``methode``
     - Principe
   * - ``"aipw"``
     - Doublement robuste : combine modèle de propension et modèle de résultat ;
       reste correct si l'un des deux est bien spécifié. **Recommandé.**
   * - ``"ipw"``
     - Repondère les individus par l'inverse de leur probabilité de recevoir le
       traitement qu'ils ont reçu (scores tronqués à [0,01 ; 0,99]).
   * - ``"appariement"``
     - Associe à chaque traité le(s) témoin(s) au score de propension le plus
       proche (caliper 0,2 é.-t. du logit). Estime l'effet chez les traités (ATT).
   * - ``"regression"``
     - Ajustement linéaire classique (référence de comparaison).

``estimand="ATE"`` : effet moyen si toute la population était traitée ;
``"ATT"`` : effet moyen chez ceux qui ont effectivement bénéficié du dispositif.
Pour un résultat binaire, l'effet est une différence de probabilités
(x100 = points de pourcentage). ``bootstrap=500`` fournit des erreurs-types
par rééchantillonnage, qui intègrent l'incertitude sur le score de propension.

Différence de différences
-------------------------

Pour un dispositif déployé à une date donnée sur certains territoires :

.. code-block:: python

   did = ca.difference_en_differences(df, resultat="acces_emploi", traite="territoire_pilote",
                                      apres="periode_post", grappes="mission_locale")
   did["effet"], did["moyennes"]

Vérifier graphiquement les **tendances parallèles** avant le déploiement
(:func:`obskit.descriptif.evolution` par groupe).

Sensibilité
-----------

.. code-block:: python

   ca.e_value(1.8, borne_ic=1.3)        # RR = 1,8 ; IC bas = 1,3

Une E-value de 3 signifie qu'un facteur non observé devrait être associé au
traitement **et** au résultat avec un rapport de risques d'au moins 3 pour
annuler l'effet estimé.
