import sys
import os
from unittest.mock import MagicMock as Mock
from setuptools_scm import get_version

MOCK_MODULES = [
	'numpy', 'numpy.testing', 'numpy.random',
	'scipy',
	'symengine', 'symengine.printing', 'symengine.lib.symengine_wrapper',
	'jitcxde_common.helpers','jitcxde_common.numerical','jitcxde_common.symbolic','jitcxde_common.transversal'
	]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

class GroupHandler_mock(object): pass
sys.modules['jitcxde_common.transversal'] = Mock(GroupHandler=GroupHandler_mock)

sys.path.insert(0,os.path.abspath("../examples"))
sys.path.insert(0,os.path.abspath("../jitcdde"))

needs_sphinx = '1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
]

source_suffix = '.rst'

master_doc = 'index'

project = u'JiTCDDE'
copyright = u'2016, Gerrit Ansmann'

release = version = get_version(root='..', relative_to=__file__)

default_role = "any"

add_function_parentheses = True

add_module_names = False

html_theme = 'nature'
pygments_style = 'colorful'
htmlhelp_basename = 'JiTCDDEdoc'

numpydoc_show_class_members = False
autodoc_member_order = 'bysource'

def on_missing_reference(app, env, node, contnode):
	if node['reftype'] == 'any':
		return contnode
	else:
		return None

def setup(app):
	app.connect('missing-reference', on_missing_reference)
