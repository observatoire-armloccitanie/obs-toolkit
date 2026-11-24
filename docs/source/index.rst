obskit - boîte à outils de l'Observatoire
=========================================

**obskit** rassemble, sous forme de fonctions Python documentées et testées, les
analyses statistiques courantes de l'Observatoire de l'ARML Occitanie : du tri à
plat d'une extraction i-milo jusqu'aux modèles de survie et aux méthodes
d'évaluation d'impact.

Chaque fonction est **généraliste** : la variable à expliquer (``cible``), les
variables explicatives, les groupes de comparaison et les pondérations sont des
paramètres. Le même code sert donc pour l'accès à l'emploi, la durée
d'accompagnement, la sortie de CEJ, la mobilité européenne…

.. code-block:: python

   import obskit as ok

   df = ok.io.lire_fichier("extraction_imilo.csv", nettoyer_noms=True)
   ok.descriptif.tableau_croise(df, "niveau_formation", "situation_sortie")

   res = ok.regression.regression_logistique(
       df, cible="acces_emploi",
       explicatives=["sexe", "age", "qpv", "niveau_formation"],
       reference={"niveau_formation": "Niveau 3 (CAP-BEP)"},
   )
   print(res.resume())

.. list-table:: Sous-modules
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Contenu
   * - :mod:`obskit.io`
     - Lecture des extractions (CSV, Excel, Parquet), export Excel multi-onglets
   * - :mod:`obskit.preparation`
     - Nettoyage, types, âges et tranches, durées, recodages, imputation
   * - :mod:`obskit.descriptif`
     - Tris à plat et croisés, « tableau 1 », statistiques pondérées, évolutions
   * - :mod:`obskit.inference`
     - Tests (χ², Fisher, Welch, Mann-Whitney, Kruskal…), IC, tailles d'effet
   * - :mod:`obskit.correlation`
     - Corrélations, V de Cramér, eta, corrélation partielle, liens avec une cible
   * - :mod:`obskit.regression`
     - Régressions linéaire, logistique, multinomiale, Poisson ; effets marginaux
   * - :mod:`obskit.survie`
     - Kaplan-Meier, log-rank, Cox, risques concurrents
   * - :mod:`obskit.causalite`
     - Score de propension, IPW, AIPW, appariement, différence de différences
   * - :mod:`obskit.factoriel`
     - ACP, AFC, ACM, CAH, typologies
   * - :mod:`obskit.ponderation`
     - Redressement par calage (raking), effet de plan
   * - :mod:`obskit.ml`
     - Modèles prédictifs (scikit-learn), segmentation
   * - :mod:`obskit.visualisation`
     - Graphiques à la charte de l'Observatoire

.. toctree::
   :maxdepth: 2
   :caption: Prise en main

   installation
   demarrage
   choisir_methode

.. toctree::
   :maxdepth: 2
   :caption: Guides thématiques

   guides/preparation
   guides/descriptif
   guides/correlation
   guides/regression
   guides/survie
   guides/causalite
   guides/factoriel
   guides/ponderation
   guides/ml
   guides/visualisation

.. toctree::
   :maxdepth: 2
   :caption: Référence

   api/index
   validation
   contribuer
