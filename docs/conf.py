
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
from importlib.util import find_spec

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

sys.path.insert(0, ROOT_DIR)
import sphinx_rtd_theme

autodoc_mock_imports = [
    dependency
    for dependency in ('tensorflow', 'librosa', 'numpy')
    if find_spec(dependency) is None
]
autodoc_member_order = 'bysource'
if autodoc_mock_imports:
    # When TensorFlow is mocked on Read the Docs, Sphinx reports mocked decorator signature warnings under the generic "autodoc" warning type; keeping this conditional preserves -W for references, rST, toctrees, and other documentation issues when the real dependencies are installed.
    suppress_warnings = ['autodoc']


# -- Project information -----------------------------------------------------

project = 'Kapre'
copyright = '2020, Keunwoo Choi, Deokjin Joo and Juho Kim'
author = 'Keunwoo Choi, Deokjin Joo and Juho Kim'

# The full version, including alpha/beta/rc tags
with open(os.path.join(ROOT_DIR, 'kapre', '__init__.py'), 'r', encoding='utf-8') as f:
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
    if not match:
        raise RuntimeError("Unable to find version string in kapre/__init__.py")
    release = match.group(1)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",  # source linkage
    "sphinx.ext.napoleon",
    # "sphinx.ext.autosummary",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# autosummary_generate = True

# autoapi_type = 'python'
# autoapi_dirs = ['../kapre']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]


master_doc = 'index'
