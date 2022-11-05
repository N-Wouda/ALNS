import datetime
import os
import sys

# -- Project information

sys.path.insert(0, os.path.abspath("../../"))

now = datetime.date.today()

project = "ALNS"
copyright = f"2019 - {now.year}, Niels Wouda and contributors"
author = "Niels Wouda and contributors"

release = version = "4.1.0"

# -- Autodoc

autoclass_content = 'class'

autodoc_member_order = 'bysource'

autodoc_default_flags = ['members']

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

exclude_patterns = ['_build', '**.ipynb_checkpoints']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
