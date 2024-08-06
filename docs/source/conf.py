import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Modularl"
copyright = "2024, Zakaria Narjis"
author = "Zakaria Narjis"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "show-inheritance": True,
# }

templates_path = ["_templates"]
exclude_patterns = []

master_doc = "index"
autodoc_member_order = "bysource"
# autodoc_mock_imports = [
#     'modularl.agents.sac.SAC',
#     'modularl.agents.td3.TD3',
#     'modularl.agents.agent',
# ]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/modulaRL_logo.svg"
# Add the project root to the system path
