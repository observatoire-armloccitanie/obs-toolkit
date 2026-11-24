Analyses factorielles et typologies
===================================

Module : :mod:`obskit.factoriel`.

ACM + CAH : construire une typologie de profils
-----------------------------------------------

.. code-block:: python

   from obskit import factoriel as fa

   actives = ["sexe", "tranche_age", "qpv", "niveau_formation", "permis_b", "dispositif"]
   acm = fa.acm(df, actives, variables_sup=["situation_sortie"], correction_benzecri=True)

   acm.valeurs_propres                     # inertie par axe (+ taux corrigés de Benzécri)
   acm.variables["contrib"]                # contributions des modalités (%)
   acm.variables["v_test"]                 # position significative si |v-test| > 2
   acm.variables["eta2"]                   # liaison variable / axe
   acm.supplementaires["modalites"]        # projection des situations de sortie
   acm.tracer(axes=(1, 2))

   cah = fa.classification(acm.individus["coord"].iloc[:, :5])   # nombre de classes automatique
   cah["n_classes"], cah["classes"].value_counts()
   fa.caracteriser_classes(df, cah["classes"], actives + ["situation_sortie"])

En ACM les taux d'inertie bruts sont toujours faibles : utiliser les taux
corrigés de Benzécri pour juger de l'importance des axes.

ACP
---

.. code-block:: python

   acp = fa.acp(indicateurs_ml, ["taux_cej", "taux_emploi", "part_qpv", "part_infra3"],
                variables_sup=["taux_chomage_zone"])
   acp.variables["coord"]                  # corrélations variables-axes
   acp.tracer()                            # cercle des corrélations
   acp.tracer(quoi="individus")

AFC
---

.. code-block:: python

   import pandas as pd
   afc = fa.afc(pd.crosstab(df["departement"], df["situation_sortie"]))
   afc.details["inertie_totale"]           # = χ² / n
   afc.tracer()                            # lignes et colonnes sur le même plan

Les conventions (coordonnées, contributions en %, cos²) sont celles de FactoMineR.
