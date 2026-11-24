import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from obskit.donnees_exemple import generer_jeunes  # noqa: E402


@pytest.fixture(scope="session")
def df():
    """Jeu de données synthétique de taille moyenne."""
    return generer_jeunes(3000, graine=123)


@pytest.fixture(scope="session")
def df_grand():
    """Jeu plus grand pour vérifier la récupération des vrais paramètres."""
    return generer_jeunes(20000, graine=7)


@pytest.fixture
def rng():
    return np.random.default_rng(2024)


@pytest.fixture(autouse=True)
def _fermer_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
