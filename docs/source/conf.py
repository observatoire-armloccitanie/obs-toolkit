"""Configuration Sphinx de la documentation obskit."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../../src"))

import obskit  # noqa: E402

project = "obskit"
author = "Observatoire ARML Occitanie"
copyright = f"{date.today().year}, {author}"
version = release = obskit.__version__
language = "fr"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]
try:
    import sphinx_copybutton  # noqa: F401

    extensions.append("sphinx_copybutton")
except ImportError:
    pass

# --- autodoc / napoleon -----------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["sklearn"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# --- thème -------------------------------------------------------------------
try:
    import pydata_sphinx_theme  # noqa: F401

    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "navigation_with_keys": True,
        "show_toc_level": 2,
        "footer_start": ["copyright"],
    }
except ImportError:  # repli sans dépendance
    html_theme = "alabaster"

html_title = f"obskit {release}"
html_static_path = ["_static"]
html_css_files = ["obskit.css"]
