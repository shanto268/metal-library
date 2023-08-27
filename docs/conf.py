# -- Path setup --------------------------------------------------------------
import sphinx_rtd_theme  # Replace with your theme's Python package
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# -- Project information -----------------------------------------------------

project = 'SQuADDS: A validated design database and simulation workflow for superconducting qubit design'
copyright = '2023, Sadman Ahmed Shanto'
author = 'Sadman Ahmed Shanto'
version = '0.0'
release = 'testing-0'
html_logo = '_static/lfl_logo.png'

# -- General configuration ---------------------------------------------------

# Sphinx extension modules
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme'  # Add this for the Read the Docs theme
]

# Source file parsers
source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master document
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# If false, no module index is generated.
html_domain_indices = True

def setup(app):
    app.add_css_file('custom.css')
