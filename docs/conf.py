import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

project = "cce-hack"
copyright = "2026, DataHacks team"
author = "DataHacks team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "pydata_sphinx_theme"
html_title = "CCE Mooring — Hackathon Docs"

autodoc_member_order = "bysource"
napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
