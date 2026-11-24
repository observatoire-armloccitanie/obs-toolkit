# Journal des modifications

## 0.1.0 - 2026-11-24

- Compatibilité **Python 3.15** déclarée et testée en intégration continue (matrice 3.10 → 3.15).
- Versions minimales des dépendances sous 3.15 alignées sur les premières versions disposant
  de roues cp315 vérifiées (numpy 2.5.2, pandas 3.1.1, scipy 1.18.1, matplotlib 3.11.2,
  scikit-learn 1.9.1, statsmodels 0.15.0).
- Les tests transforment en erreur tout `DeprecationWarning` émis par obskit.

## Première version.

- `io` : lecture CSV/Excel/Parquet avec détection d'encodage et de séparateur, export Excel multi-onglets.
- `preparation` : noms de colonnes, types, âges et tranches, durées, recodages, regroupement des modalités rares, imputation.
- `descriptif` : panorama des variables, tris à plat et croisés (pondérés, IC), « tableau 1 », évolutions temporelles.
- `inference` : IC de proportions/moyennes, χ²/Fisher, comparaisons de groupes avec tailles d'effet, corrections multiples, taille d'échantillon.
- `correlation` : matrices de corrélation avec p-valeurs, corrélation partielle, V de Cramér, eta, associations avec une cible.
- `regression` : linéaire, logistique, multinomiale, Poisson ; erreurs robustes et par grappes ; effets marginaux moyens ; tests de Wald et RV ; VIF ; sélection pas à pas.
- `survie` : Kaplan-Meier, log-rank, Cox (Efron/Breslow, stratification, prédiction), test de proportionnalité, incidence cumulée en risques concurrents.
- `causalite` : score de propension, équilibre (SMD), IPW, AIPW, appariement, différence de différences, E-value.
- `factoriel` : ACP, AFC, ACM (+ Benzécri, variables supplémentaires), CAH, caractérisation des classes.
- `ponderation` : raking (borné), effet de plan.
- `ml` : modèles prédictifs scikit-learn évalués (test + validation croisée), importance par permutation, k-means.
- `visualisation` : barres, histogrammes, boîtes, courbes de survie, forest plot, heatmap, plans factoriels, ROC, évolutions.
