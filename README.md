# obskit - boîte à outils statistiques de l'Observatoire ARML Occitanie

Kit Python de fonctions **réutilisables, documentées et testées** pour les études de
l'Observatoire des Missions Locales d'Occitanie : statistiques descriptives, tests,
corrélations, régressions (linéaire, logistique, multinomiale, Poisson), analyse de survie
(Kaplan-Meier, Cox, risques concurrents), évaluation d'impact (score de propension, AIPW,
différence de différences), analyses factorielles (ACP, AFC, ACM, CAH), redressement
d'enquêtes, apprentissage automatique et graphiques à la charte.

Toutes les fonctions sont **généralistes** : la variable cible, les explicatives, les groupes
et les pondérations sont des paramètres.

```python
import obskit as ok

df = ok.io.lire_fichier("extraction_imilo.csv", nettoyer_noms=True)

ok.descriptif.tableau_croise(df, "niveau_formation", "situation_sortie")

res = ok.regression.regression_logistique(
    df, cible="acces_emploi",
    explicatives=["sexe", "age", "qpv", "niveau_formation"],
    reference={"niveau_formation": "Niveau 3 (CAP-BEP)"},
)
print(res.resume())
ok.regression.effets_marginaux_moyens(res)

km = ok.survie.kaplan_meier(df, "duree", "evenement", groupe="qpv")
cox = ok.survie.modele_cox(df, "duree", "evenement", ["sexe", "age", "qpv"])
```

## Installation

Python 3.10 à 3.15 (testé en intégration continue sur 3.10, 3.12, 3.13, 3.14 et 3.15).

```bash
git clone <url-du-depot> ressources_python
cd ressources_python
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[ml]"            # ".[dev]" pour les tests, ".[docs]" pour la documentation
```

## Organisation

| Sous-module                | Contenu                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `obskit.io`              | Lecture CSV/Excel/Parquet (encodage et séparateur détectés), export Excel multi-onglets           |
| `obskit.preparation`     | Noms de colonnes, types, âges et tranches, durées, recodages, modalités rares, imputation         |
| `obskit.descriptif`      | Panorama, tris à plat et croisés (pondérés, IC), « tableau 1 », évolutions temporelles        |
| `obskit.inference`       | IC, χ²/Fisher, comparaisons de groupes avec tailles d'effet, corrections multiples                 |
| `obskit.correlation`     | Corrélations avec p-valeurs, V de Cramér, eta, corrélation partielle, liens avec une cible        |
| `obskit.regression`      | Linéaire, logistique, multinomiale, Poisson ; erreurs robustes/cluster ; effets marginaux           |
| `obskit.survie`          | Kaplan-Meier, log-rank, Cox (Efron, strates, prédiction), test de proportionnalité, Aalen-Johansen |
| `obskit.causalite`       | Score de propension, SMD, IPW, AIPW, appariement, différence de différences, E-value               |
| `obskit.factoriel`       | ACP, AFC, ACM (Benzécri, supplémentaires), CAH, caractérisation des classes                       |
| `obskit.ponderation`     | Calage sur marges (raking), effet de plan                                                            |
| `obskit.ml`              | Modèles prédictifs scikit-learn évalués, importance par permutation, k-means                     |
| `obskit.visualisation`   | Barres, survie, forest plot, heatmap, plans factoriels, ROC, évolutions                             |
| `obskit.donnees_exemple` | Données synthétiques réalistes pour s'exercer et tester                                           |

```
obs-toolkit/
├── src/obskit/          # le package
├── tests/               # tests pytest (validation contre scipy, scikit-learn, statsmodels, lifelines)
├── docs/                # documentation Sphinx (publiée sur GitHub Pages)
├── examples/            # script d'analyse complet
└── .github/workflows/   # intégration continue : tests + publication de la documentation
```

## Tests

```bash
pip install -e ".[dev]"
pytest                               # ~120 tests
pytest --doctest-modules src/obskit  # exemples des docstrings
```

Les estimateurs sont implémentés avec numpy/scipy et vérifiés contre des résultats de
référence (formules exactes, scipy, scikit-learn, et statsmodels/lifelines quand ils sont
installés) ainsi que par la récupération des vrais paramètres sur données simulées.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

Sur GitHub : *Settings → Pages → Source : GitHub Actions*. Le workflow
`.github/workflows/docs.yml` publie la documentation à chaque `push` sur `main`.

## Données

Aucune donnée individuelle ne doit être versionnée (le `.gitignore` exclut les `.csv`,
`.xlsx`, `data/`…). Les exemples utilisent `obskit.donnees_exemple.generer_jeunes()`.

## Licence

MIT - voir `LICENSE`.
