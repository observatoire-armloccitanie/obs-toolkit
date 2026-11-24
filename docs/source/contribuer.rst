Contribuer
==========

Environnement de développement
------------------------------

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev,docs]"

Conventions
-----------

* **Noms en français**, en ``snake_case`` sans accents (``tableau_frequences``, ``modele_cox``).
* **Fonctions généralistes** : la variable cible et les explicatives sont toujours des
  paramètres (``cible=``, ``explicatives=``, ``groupe=``…), jamais codées en dur.
* **Docstrings au format NumPy** (``Parameters``, ``Returns``, ``Examples``, ``Notes``) :
  elles alimentent directement la :doc:`référence de l'API <api/index>`.
* Les fonctions renvoient des objets exploitables (DataFrame ou objet ``Resultat…``).
* Aucune donnée réelle dans le dépôt : utiliser :mod:`obskit.donnees_exemple`.

Avant de pousser
----------------

.. code-block:: bash

   ruff check src tests
   pytest
   pytest --doctest-modules src/obskit
   sphinx-build -b html docs/source docs/build/html

Ajouter une fonction
--------------------

#. L'écrire dans le sous-module thématique et l'ajouter à ``__all__``.
#. Écrire un test comparant le résultat à une valeur connue ou à une bibliothèque de référence.
#. Ajouter si besoin un exemple dans le guide correspondant (``docs/source/guides/``).
#. Mettre à jour ``CHANGELOG.md``.
