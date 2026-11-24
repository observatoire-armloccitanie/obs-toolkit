"""obskit - boîte à outils statistiques de l'Observatoire ARML Occitanie.

**Sous-modules**

* :mod:`obskit.io` - lecture des extractions (CSV/Excel) et export Excel ;
* :mod:`obskit.preparation` - nettoyage, recodages, tranches d'âge, durées ;
* :mod:`obskit.descriptif` - tris à plat, tris croisés, tableau 1, évolutions ;
* :mod:`obskit.inference` - tests statistiques, intervalles de confiance ;
* :mod:`obskit.correlation` - corrélations et associations tous types ;
* :mod:`obskit.regression` - régressions linéaire, logistique, multinomiale, Poisson ;
* :mod:`obskit.survie` - Kaplan-Meier, log-rank, Cox, risques concurrents ;
* :mod:`obskit.causalite` - score de propension, IPW, AIPW, appariement, DiD ;
* :mod:`obskit.factoriel` - ACP, AFC, ACM, CAH et typologies ;
* :mod:`obskit.ponderation` - redressement (raking), effet de plan ;
* :mod:`obskit.ml` - modèles prédictifs scikit-learn, segmentation ;
* :mod:`obskit.visualisation` - graphiques à la charte de l'Observatoire ;
* :mod:`obskit.donnees_exemple` - données synthétiques pour s'exercer.

**Exemple**

>>> import obskit as ok
>>> df = ok.donnees_exemple.generer_jeunes(1000)
>>> res = ok.regression.regression_logistique(df, "acces_emploi", ["sexe", "qpv", "age"])
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_SOUS_MODULES = (
    "io", "preparation", "descriptif", "inference", "correlation", "regression", "survie",
    "causalite", "factoriel", "ponderation", "ml", "visualisation", "donnees_exemple",
)

__all__ = list(_SOUS_MODULES) + ["__version__"]


def __getattr__(nom: str):
    """Import paresseux des sous-modules (``obskit.survie`` n'est chargé qu'à l'usage)."""
    if nom in _SOUS_MODULES:
        module = importlib.import_module(f".{nom}", __name__)
        globals()[nom] = module
        return module
    raise AttributeError(f"module 'obskit' has no attribute '{nom}'")


def __dir__():
    return sorted(list(globals()) + list(_SOUS_MODULES))
