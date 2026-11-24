"""Jeux de données synthétiques pour la documentation, les tutoriels et les tests.

Les données sont **entièrement simulées** : elles imitent la structure d'une
extraction i-milo (jeunes accompagnés par les Missions Locales d'Occitanie)
sans contenir aucune donnée réelle. Les effets simulés sont connus
(:data:`PARAMETRES_VRAIS`), ce qui permet de vérifier que les méthodes
retrouvent bien les « vrais » paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generer_jeunes", "PARAMETRES_VRAIS", "DEPARTEMENTS_OCCITANIE", "NIVEAUX_FORMATION"]

DEPARTEMENTS_OCCITANIE = {
    "09": "Ariège", "11": "Aude", "12": "Aveyron", "30": "Gard", "31": "Haute-Garonne",
    "32": "Gers", "34": "Hérault", "46": "Lot", "48": "Lozère", "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales", "81": "Tarn", "82": "Tarn-et-Garonne",
}
_POIDS_DEP = np.array([2, 5, 3, 10, 22, 2, 18, 2, 1, 3, 7, 5, 4], dtype=float)

NIVEAUX_FORMATION = ["Infra 3", "Niveau 3 (CAP-BEP)", "Niveau 4 (Bac)", "Niveau 5 et +"]

#: Paramètres utilisés pour simuler les données (échelle du prédicteur linéaire).
PARAMETRES_VRAIS = {
    # régression logistique de `acces_emploi`
    "logistique": {
        "const": -2.2, "sexe[Homme]": 0.25, "qpv[True]": -0.5, "age": 0.08,
        "niveau_formation[Infra 3]": -0.7, "niveau_formation[Niveau 4 (Bac)]": 0.4,
        "niveau_formation[Niveau 5 et +]": 0.8, "permis_b[True]": 0.6,
        "dispositif[CEJ]": 0.3, "dispositif[Autre]": -0.2,
    },
    # modèle de Cox (risque instantané d'accès à l'emploi)
    "cox": {
        "sexe[Homme]": 0.15, "qpv[True]": -0.4, "age": 0.05,
        "niveau_formation[Infra 3]": -0.5, "niveau_formation[Niveau 4 (Bac)]": 0.3,
        "niveau_formation[Niveau 5 et +]": 0.6, "permis_b[True]": 0.5,
    },
    # régression linéaire de `score_insertion`
    "lineaire": {
        "const": 40.0, "sexe[Homme]": 1.5, "qpv[True]": -4.0, "age": 0.8,
        "niveau_formation[Infra 3]": -6.0, "niveau_formation[Niveau 4 (Bac)]": 3.0,
        "niveau_formation[Niveau 5 et +]": 7.0, "permis_b[True]": 5.0, "sigma": 8.0,
    },
    # Poisson de `nb_entretiens` avec exposition `duree_accompagnement_mois`
    "poisson": {"const": 0.2, "dispositif[CEJ]": 0.5, "dispositif[Autre]": -0.3, "qpv[True]": 0.1},
}


def generer_jeunes(n: int = 2000, graine: int = 0, taux_manquants: float = 0.0) -> pd.DataFrame:
    """Génère un jeu de données synthétique de jeunes accompagnés.

    Parameters
    ----------
    n : int
        Nombre de jeunes.
    graine : int
        Graine aléatoire (reproductibilité).
    taux_manquants : float
        Part de valeurs manquantes injectées dans ``niveau_formation`` et ``permis_b``.

    Returns
    -------
    DataFrame
        Colonnes : ``id_jeune``, ``mission_locale``, ``departement``, ``sexe``, ``age``,
        ``qpv``, ``zrr``, ``niveau_formation``, ``permis_b``, ``dispositif``,
        ``date_entree``, ``date_acces_emploi``, ``duree_emploi_mois``,
        ``evenement_emploi``, ``acces_emploi``, ``score_insertion``, ``nb_entretiens``,
        ``duree_accompagnement_mois``, ``situation_sortie``, ``poids``.

    Examples
    --------
    >>> from obskit.donnees_exemple import generer_jeunes
    >>> df = generer_jeunes(500, graine=1)
    >>> df.shape[0]
    500
    """
    rng = np.random.default_rng(graine)
    deps = rng.choice(list(DEPARTEMENTS_OCCITANIE), size=n, p=_POIDS_DEP / _POIDS_DEP.sum())
    ml = np.array([f"ML{d}-{rng.integers(1, 4)}" for d in deps])
    sexe = rng.choice(["Femme", "Homme"], size=n, p=[0.48, 0.52])
    age = rng.integers(16, 26, size=n)
    qpv = rng.random(n) < 0.22
    zrr = (~qpv) & (rng.random(n) < 0.12)
    niv = rng.choice(NIVEAUX_FORMATION, size=n, p=[0.25, 0.35, 0.28, 0.12])
    permis = rng.random(n) < (0.15 + 0.03 * (age - 16))
    dispositif = rng.choice(["CEJ", "PACEA", "Autre"], size=n, p=[0.45, 0.35, 0.20])
    debut = pd.Timestamp("2023-01-01")
    date_entree = debut + pd.to_timedelta(rng.integers(0, 3 * 365, size=n), unit="D")

    def lin(params, extra=None):
        eta = np.full(n, params.get("const", 0.0))
        eta += params.get("sexe[Homme]", 0) * (sexe == "Homme")
        eta += params.get("qpv[True]", 0) * qpv
        eta += params.get("age", 0) * age
        for m in NIVEAUX_FORMATION:
            eta += params.get(f"niveau_formation[{m}]", 0) * (niv == m)
        eta += params.get("permis_b[True]", 0) * permis
        eta += params.get("dispositif[CEJ]", 0) * (dispositif == "CEJ")
        eta += params.get("dispositif[Autre]", 0) * (dispositif == "Autre")
        return eta

    # logistique
    acces = rng.random(n) < 1 / (1 + np.exp(-lin(PARAMETRES_VRAIS["logistique"])))

    # survie : Weibull à risques proportionnels
    pc = PARAMETRES_VRAIS["cox"]
    eta_cox = lin(pc)
    k, lam = 1.2, 0.02
    u = rng.random(n)
    t_evt = (-np.log(u) / (lam * np.exp(eta_cox))) ** (1 / k)
    fin_obs = pd.Timestamp("2026-06-30")
    c_admin = (fin_obs - date_entree).days / (365.25 / 12)
    c_perdu = rng.exponential(60, size=n)
    c = np.minimum(c_admin.to_numpy(), c_perdu)
    duree = np.minimum(t_evt, c)
    evt = (t_evt <= c).astype(int)
    date_emploi = pd.Series(date_entree + pd.to_timedelta(duree * 365.25 / 12, unit="D"))
    date_emploi[evt == 0] = pd.NaT

    # linéaire
    plin = PARAMETRES_VRAIS["lineaire"]
    score = lin({k_: v for k_, v in plin.items() if k_ != "sigma"}) + rng.normal(0, plin["sigma"], n)

    # Poisson avec exposition
    expo = np.round(rng.uniform(1, 24, size=n), 1)
    pp = PARAMETRES_VRAIS["poisson"]
    taux = np.exp(lin(pp))
    nb_ent = rng.poisson(taux * expo)

    # multinomiale : situation à la sortie
    eta_emp = 0.3 + 0.5 * permis - 0.4 * qpv + 0.4 * (niv == "Niveau 5 et +")
    eta_form = -0.2 + 0.4 * (age < 19) + 0.3 * (niv == "Infra 3")
    eta_ss = -0.6 + 0.5 * qpv - 0.3 * permis
    P = np.exp(np.column_stack([np.zeros(n), eta_emp, eta_form, eta_ss]))
    P /= P.sum(axis=1, keepdims=True)
    issues = np.array(["Autre", "Emploi", "Formation", "Sans solution"])
    sortie = issues[(rng.random(n)[:, None] > np.cumsum(P, axis=1)).sum(axis=1)]

    poids = np.round(rng.uniform(0.5, 2.0, size=n), 3)
    df = pd.DataFrame({
        "id_jeune": np.arange(1, n + 1),
        "mission_locale": ml,
        "departement": deps,
        "sexe": sexe,
        "age": age,
        "qpv": qpv,
        "zrr": zrr,
        "niveau_formation": pd.Categorical(niv, categories=NIVEAUX_FORMATION, ordered=True),
        "permis_b": permis,
        "dispositif": dispositif,
        "date_entree": date_entree,
        "date_acces_emploi": date_emploi.to_numpy(),
        "duree_emploi_mois": np.round(duree, 2),
        "evenement_emploi": evt,
        "acces_emploi": acces.astype(int),
        "score_insertion": np.round(score, 2),
        "nb_entretiens": nb_ent,
        "duree_accompagnement_mois": expo,
        "situation_sortie": sortie,
        "poids": poids,
    })
    if taux_manquants > 0:
        for col in ("niveau_formation", "permis_b"):
            m = rng.random(n) < taux_manquants
            df[col] = df[col].astype(object)
            df.loc[m, col] = np.nan
        df["niveau_formation"] = pd.Categorical(df["niveau_formation"], categories=NIVEAUX_FORMATION,
                                                ordered=True)
    return df
