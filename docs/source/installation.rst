Installation
============

Prérequis : Python 3.10 à 3.15.

Sous **Python 3.15**, ``pip`` installe automatiquement des versions des
dépendances qui publient des roues précompilées pour cette version (numpy ≥ 2.5.2,
pandas ≥ 3.1.1, scipy ≥ 1.18.1, matplotlib ≥ 3.11.2, scikit-learn ≥ 1.9.1) :
aucune compilation n'est nécessaire sous Windows.

.. code-block:: bash

   py -3.15 -m venv .venv          # lanceur Windows : choisir explicitement 3.15

Depuis le dépôt (recommandé pour l'équipe)
------------------------------------------

.. code-block:: bash

   git clone <url-du-depot> ressources_python
   cd ressources_python
   python -m venv .venv
   .venv\Scripts\activate        # Windows ; source .venv/bin/activate ailleurs
   pip install -e ".[ml]"

L'option ``-e`` (installation « éditable ») permet de modifier le code du kit
et d'en profiter immédiatement dans ses scripts et notebooks.

Options
-------

* ``pip install -e .`` : cœur du kit (numpy, pandas, scipy, matplotlib, openpyxl) ;
* ``pip install -e ".[ml]"`` : ajoute scikit-learn pour :mod:`obskit.ml` ;
* ``pip install -e ".[dev]"`` : outils de test (pytest, ruff) et bibliothèques
  de référence (statsmodels, lifelines) utilisées pour la validation croisée ;
* ``pip install -e ".[docs]"`` : construction de cette documentation.

Construire la documentation localement
--------------------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html

Ouvrir ensuite ``docs/build/html/index.html``. Sur GitHub, la documentation est
publiée automatiquement sur GitHub Pages à chaque ``push`` sur ``main``
(workflow ``.github/workflows/docs.yml`` ; activer *Settings → Pages → Source :
GitHub Actions* dans le dépôt).
