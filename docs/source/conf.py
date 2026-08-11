# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

#
from importlib.metadata import version

# Make sure the project root (containing the `mcdc/` package) is importable
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXTENSIONS_ROOT = os.path.join(HERE, "_ext")
if EXTENSIONS_ROOT not in sys.path:
    sys.path.insert(0, EXTENSIONS_ROOT)

# -- Project information -----------------------------------------------------
project = "MC/DC"
copyright = "2023-2026, Center for Exascale Monte Carlo Neutron Transport (CEMeNT), Center for Advancing the Radiation Resilience of Electronics (CARRE), and MC/DC contributors"

# The full version, including alpha/beta/rc tags
release = version("mcdc")

# The short X.Y version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "simulation_members",
]
autosummary_generate = True
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The PyData Sphinx Theme provides the site-wide header, section navigation,
# in-page table of contents, search, and light/dark mode.
html_theme = "pydata_sphinx_theme"
html_title = "MC/DC Documentation"
html_logo = "../../assets/mcdc-logo.svg"
html_favicon = "../../assets/mcdc-favicon.svg"

# Read the Docs provides the active version slug during hosted builds. Use the
# development docs as the local-build default.
docs_version = os.environ.get("READTHEDOCS_VERSION", "dev")
switcher_json_url = (
    "https://mcdc.readthedocs.io/en/dev/_static/switcher.json"
    if os.environ.get("READTHEDOCS") == "True"
    else "/_static/switcher.json"
)

html_theme_options = {
    "navbar_align": "left",
    "navbar_end": ["theme-toggle", "version-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "switcher": {
        "json_url": switcher_json_url,
        "version_match": docs_version,
    },
    "show_version_warning_banner": docs_version != "stable",
    # Hosted builds share one version list from the development documentation.
    # Local builds use the copied static file when served from the HTML output root.
    "check_switcher": False,
    "logo": {
        "alt_text": "MC/DC Documentation - Home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mcdc-project/mcdc",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mcdc/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
}

html_context = {
    "github_user": "mcdc-project",
    "github_repo": "mcdc",
    "github_version": "dev",
    "doc_path": "docs/source",
}

# The home page already provides purpose-based routing. Section pages use the
# theme's default collapsible primary sidebar.
html_sidebars = {
    "index": [],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["theme-toggle.js"]
