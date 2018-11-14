from setuptools import setup
from io import open

requirements = [
	'jitcxde_common>1.3',
	'symengine>=0.3.1.dev0',
	'numpy'
]

setup(
	name = 'jitcdde',
	description = 'Just-in-Time Compilation for Delay Differential Equations',
	long_description = open('README.rst', encoding='utf8').read(),
	author = 'Gerrit Ansmann',
	author_email = 'gansmann@uni-bonn.de',
	url = 'http://github.com/neurophysik/jitcdde',
	python_requires=">=3.3",
	packages = ['jitcdde'],
	package_data = {'jitcdde': ['jitced_template.c']},
	include_package_data = True,
	install_requires = requirements,
	setup_requires = ['setuptools_scm'],
	use_scm_version = {'write_to': 'jitcdde/version.py'},
	classifiers = [
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: BSD License',
		'Operating System :: POSIX',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering :: Mathematics',
		],
)

