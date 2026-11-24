# Contribuer au kit

## Installer l'environnement de développement

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate sous Linux/macOS)
pip install -e ".[dev,docs]"
```

## Conventions

- **Noms en français**, en `snake_case` sans accents (`tableau_frequences`, `modele_cox`).
- **Fonctions généralistes** : la variable cible et les explicatives sont toujours des paramètres
  (`cible=`, `explicatives=`, `groupe=`…), jamais codées en dur.
- **Docstrings au format NumPy** (sections `Parameters`, `Returns`, `Examples`, `Notes`) :
  elles alimentent directement la documentation Sphinx.
- Chaque fonction renvoie un objet exploitable (DataFrame ou objet `Resultat…`) plutôt que d'imprimer.
- Pas de données réelles dans le dépôt : utiliser `obskit.donnees_exemple` pour les exemples et tests.

## Avant de pousser

```bash
ruff check src tests            # style
pytest                          # tests (dont validation croisée statsmodels/lifelines)
pytest --doctest-modules src    # exemples des docstrings
sphinx-build -b html docs/source docs/build/html   # documentation
```

## Ajouter une fonction

1. L'écrire dans le sous-module thématique adéquat et l'ajouter à `__all__`.
2. Écrire au moins un test qui compare le résultat à une valeur connue ou à une bibliothèque de référence.
3. Si besoin, ajouter un exemple dans le guide correspondant (`docs/source/guides/`).
4. Noter la nouveauté dans `CHANGELOG.md`.
