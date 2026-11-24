Lire et préparer les données
============================

Modules : :mod:`obskit.io`, :mod:`obskit.preparation`.

Lire une extraction
-------------------

:func:`~obskit.io.lire_fichier` détecte le format, l'encodage (UTF-8,
Windows-1252…) et le séparateur ; pour un CSV en ``;`` elle considère la
virgule comme séparateur décimal (convention française).

.. code-block:: python

   from obskit import io, preparation as prep

   df = io.lire_fichier("export_obiee.csv", nettoyer_noms=True,
                        dtype={"departement": str})   # garder le zéro de "09"
   df = io.lire_fichier("tableau.xlsx", sheet_name="Données", skiprows=2)

``nettoyer_noms=True`` transforme « Date d'entrée en CEJ » en
``date_d_entree_en_cej`` : plus simple à manipuler en Python.

Diagnostiquer
-------------

.. code-block:: python

   prep.detecter_types(df)                      # identifiant, binaire, catégorielle, numérique, date
   from obskit import descriptif
   descriptif.valeurs_manquantes(df, seuil=1)   # colonnes avec plus de 1 % de manquants

Dates, âges et durées
---------------------

.. code-block:: python

   df = prep.convertir_dates(df, ["date_naissance", "date_entree", "date_sortie"])
   df["age_entree"] = prep.calculer_age(df["date_naissance"], df["date_entree"])
   df["tranche_age"] = prep.tranches_age(df["age_entree"])        # 16-17, 18-21, 22-25, 26 et +
   df["duree_mois"] = prep.duree_entre(df["date_entree"], df["date_sortie"], unite="mois")

Les tranches sont des catégories **ordonnées** : les tableaux et graphiques
respectent automatiquement l'ordre des âges.

Recoder
-------

.. code-block:: python

   df["situation"] = prep.recoder(
       df["situation_detaillee"],
       {("CDI", "CDD de 6 mois et plus"): "Emploi durable",
        ("CDD de moins de 6 mois", "Intérim"): "Emploi temporaire",
        ("Formation qualifiante", "Formation pré-qualifiante"): "Formation"},
       autre="Autre situation",
       ordre=["Emploi durable", "Emploi temporaire", "Formation", "Autre situation"],
   )
   df["mission_locale"] = prep.regrouper_modalites_rares(df["mission_locale"], seuil=0.02)
   df["qpv"] = prep.convertir_binaire(df["qpv"])        # "Oui"/"Non" -> 1/0

Valeurs manquantes
------------------

:func:`~obskit.preparation.imputer` remplace par défaut les manquants
numériques par la médiane et crée une modalité « Non renseigné » pour les
variables qualitatives - le non-renseignement est souvent une information en
soi (et évite d'exclure des lignes dans les modèles).

.. code-block:: python

   df = prep.imputer(df)                                     # stratégie automatique
   df = prep.imputer(df, {"niveau_formation": "non_renseigne", "age": "mediane"})
