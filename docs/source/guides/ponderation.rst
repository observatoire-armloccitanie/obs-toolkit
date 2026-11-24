Pondération et redressement
===========================

Module : :mod:`obskit.ponderation`.

Une enquête (satisfaction, mobilité européenne…) n'a jamais la même structure
que la population des jeunes accompagnés. Le **calage sur marges** (raking)
ajuste les poids pour que l'échantillon reproduise les répartitions connues
via i-milo.

.. code-block:: python

   from obskit import ponderation as po, descriptif as ds

   marges = {
       "sexe": {"Femme": 0.49, "Homme": 0.51},
       "tranche_age": {"16-17 ans": 0.12, "18-21 ans": 0.52, "22-25 ans": 0.36},
       "departement": effectifs_par_departement,          # dict, effectifs ou proportions
   }
   po.comparer_marges(enquete, marges)                     # écarts avant redressement
   enquete["poids"] = po.redressement_raking(enquete, marges, bornes=(0.3, 3))
   po.comparer_marges(enquete, marges, poids="poids")      # écarts ~ 0
   po.effet_plan(enquete["poids"])                         # perte de précision (deff de Kish)

   ds.tableau_frequences(enquete, "satisfaction", poids="poids")

``total=`` permet d'obtenir des poids d'extrapolation (somme des poids = taille
de la population). Les bornes limitent les poids extrêmes au prix d'un calage
approché.
