import datetime
import os
import sys

import tomli

# -- Project information

sys.path.insert(0, os.path.abspath("../../"))

now = datetime.date.today()

project = "ALNS"
copyright = f"2019 - {now.year}, Niels Wouda and contributors"
author = "Niels Wouda and contributors"

with open("../../pyproject.toml", "rb") as fh:
    pyproj = tomli.load(fh)
    release = version = pyproj["tool"]["poetry"]["version"]

# -- Autodoc

autoclass_content = "class"

autodoc_member_order = "bysource"

autodoc_default_flags = ["members"]

# -- Numpydoc

numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False

# -- nbsphinx

nbsphinx_execute = "never"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "numpydoc",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ('https://numpy.org/doc/stable/', None),
    "matplotlib": ('https://matplotlib.org/stable/', None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
