Graphiques
==========

Module : :mod:`obskit.visualisation`.

.. code-block:: python

   from obskit import visualisation as vz
   vz.appliquer_style()        # charte : palette, grille discrète, titres à gauche

Toutes les fonctions renvoient ``(fig, ax)`` et acceptent ``ax=`` pour composer
des figures à plusieurs panneaux.

.. code-block:: python

   tf = ds.tableau_frequences(df, "niveau_formation", total=False)
   fig, ax = vz.diagramme_barres(tf["pct"], format_etiquette="{:.1f} %",
                                 titre="Niveau de formation à l'entrée",
                                 source="Source : i-milo, traitement Observatoire ARML Occitanie")
   vz.enregistrer(fig, "sorties/niveau_formation", formats=("png", "svg"))

   vz.diagramme_barres(ds.tableau_croise(df, "niveau_formation", "situation_sortie", marges=False))
   km.tracer(incidence=True)                      # courbes de Kaplan-Meier
   vz.forest_plot(res)                            # OR / HR avec IC (échelle log)
   vz.forest_plot(lin, exposant=False)            # coefficients d'une régression linéaire
   vz.heatmap_correlation(matrice)
   vz.courbe_roc(res.y, res.valeurs_ajustees)
   vz.evolution_temporelle(ds.evolution(df, "date_entree", "M", par="dispositif"))
   acm.tracer()                                   # plan factoriel

Règles de la charte
-------------------

* une couleur par entité, attribuée dans un **ordre fixe** (:data:`~obskit.visualisation.PALETTE`) ;
* palette séquentielle à une teinte pour les intensités, divergente (bleu–gris–rouge) pour les écarts signés ;
* jamais deux axes des ordonnées : faire deux graphiques ;
* légende dès deux séries ; libellés de valeurs sélectifs ;
* toujours indiquer la source et le champ sous le graphique (``source=``).

Pour adapter la charte à une autre identité graphique, modifier
``obskit.visualisation.PALETTE`` et ``SEQUENTIELLE`` avant de tracer.
