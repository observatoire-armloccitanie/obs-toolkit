Quoi pour quoi ?
=====================================

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Question
     - Méthode
     - Fonction
   * - Comment se répartissent les jeunes selon une variable ?
     - Tri à plat, IC de Wilson
     - :func:`~obskit.descriptif.tableau_frequences`
   * - Deux variables qualitatives sont-elles liées ?
     - Tri croisé, χ² (Fisher si petits effectifs), V de Cramér
     - :func:`~obskit.descriptif.tableau_croise`
   * - Une moyenne diffère-t-elle entre groupes ?
     - t de Welch / ANOVA de Welch ou Mann-Whitney / Kruskal-Wallis
     - :func:`~obskit.inference.comparer_groupes`
   * - Deux taux diffèrent-ils (QPV vs hors QPV) ?
     - Test z de deux proportions
     - :func:`~obskit.inference.comparer_proportions`
   * - Quelles variables sont liées à ma cible ?
     - Tests bivariés + mesures d'association, correction de Holm
     - :func:`~obskit.correlation.associations_avec_cible`
   * - Effet d'une variable toutes choses égales par ailleurs (cible binaire) ?
     - Régression logistique, OR et effets marginaux moyens
     - :func:`~obskit.regression.regression_logistique`
   * - … cible quantitative ?
     - Régression linéaire (erreurs robustes HC3)
     - :func:`~obskit.regression.regression_lineaire`
   * - … cible à plusieurs modalités (situation à la sortie) ?
     - Logistique multinomiale (RRR)
     - :func:`~obskit.regression.regression_logistique_multinomiale`
   * - … cible de comptage (nombre d'entretiens, de PMSMP) ?
     - Régression de Poisson avec exposition
     - :func:`~obskit.regression.regression_poisson`
   * - En combien de temps l'événement survient-il ?
     - Kaplan-Meier, médiane, log-rank
     - :func:`~obskit.survie.kaplan_meier`
   * - Quels facteurs accélèrent / retardent l'événement ?
     - Modèle de Cox (HR)
     - :func:`~obskit.survie.modele_cox`
   * - Plusieurs issues concurrentes (emploi, formation, abandon) ?
     - Incidence cumulée d'Aalen-Johansen
     - :func:`~obskit.survie.incidence_cumulee`
   * - Quel est l'effet d'un dispositif (sans tirage au sort) ?
     - AIPW, IPW, appariement sur score de propension
     - :func:`~obskit.causalite.estimer_effet`
   * - Effet d'un déploiement territorial daté ?
     - Différence de différences
     - :func:`~obskit.causalite.difference_en_differences`
   * - Construire une typologie de profils ?
     - ACM + CAH + caractérisation
     - :func:`~obskit.factoriel.acm`, :func:`~obskit.factoriel.classification`
   * - Résumer des indicateurs quantitatifs corrélés ?
     - ACP
     - :func:`~obskit.factoriel.acp`
   * - Redresser une enquête sur la structure connue de la population ?
     - Raking
     - :func:`~obskit.ponderation.redressement_raking`
   * - Prédire au mieux (repérage de jeunes à risque) ?
     - Gradient boosting, validation croisée
     - :func:`~obskit.ml.modele_predictif`

Points de vigilance
-------------------

* **Gros effectifs** : avec des dizaines de milliers de jeunes, tout devient
  « significatif ». Regarder les **tailles d'effet** (V de Cramér, d de Cohen,
  effets marginaux) plutôt que les seules p-valeurs.
* **Odds ratios** : ils ne se lisent pas comme des rapports de probabilités
  quand l'événement est fréquent (> 10 %) et ne se comparent pas entre modèles ;
  préférer les effets marginaux moyens pour la communication.
* **Censure** : ne jamais calculer une durée moyenne en excluant les jeunes
  encore accompagnés ; utiliser Kaplan-Meier (médiane, RMST).
* **Causalité** : une association, même ajustée, n'est pas un effet. Les
  méthodes de :mod:`obskit.causalite` reposent sur l'absence de facteur de
  confusion non observé : le discuter et calculer une :func:`~obskit.causalite.e_value`.
* **Secret statistique** : avant diffusion, masquer les cellules de faible
  effectif, selon les règles de diffusion applicables aux données i-milo.
