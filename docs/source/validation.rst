Validation des calculs
======================

Les estimateurs sont implémentés directement avec numpy/scipy (aucune
dépendance lourde), et **vérifiés par les tests** (``tests/``) contre des
résultats de référence :

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Méthode
     - Référence de comparaison
   * - MCO, erreurs HC1/HC3, cluster
     - Formules matricielles exactes ; ``statsmodels`` (OLS)
   * - Logistique, Poisson
     - ``scikit-learn`` (sans pénalisation) ; ``statsmodels`` (Logit, GLM)
   * - Logistique multinomiale
     - ``scikit-learn`` ; ``statsmodels`` (MNLogit)
   * - Kaplan-Meier, log-rank
     - ``scipy.stats.ecdf`` et ``scipy.stats.logrank`` ; ``lifelines``
   * - Cox (Efron / Breslow)
     - Maximisation directe de la vraisemblance partielle ; ``lifelines``
   * - Concordance de Harrell
     - Calcul par force brute sur toutes les paires
   * - IC de Wilson / Clopper-Pearson
     - ``scipy.stats.binomtest``
   * - ACP, AFC, ACM
     - Propriétés exactes (valeurs propres de la matrice de corrélation,
       inertie = χ²/n, somme des valeurs propres = (J − Q)/Q)
   * - Régressions, Cox, causalité
     - Récupération des **vrais paramètres** sur données simulées
       (:data:`obskit.donnees_exemple.PARAMETRES_VRAIS`)

Les tests de comparaison avec ``statsmodels`` et ``lifelines`` s'exécutent
automatiquement lorsque ces bibliothèques sont installées
(``pip install -e ".[dev]"``, et systématiquement dans l'intégration continue).
