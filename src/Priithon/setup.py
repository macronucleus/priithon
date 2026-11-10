"""
Usage:
    python setup.py [py2app/sdist/install]

The code was tested with Miniconda verion of python
"""
import os #, sys, stat
from setuptools import setup

#pyversion = sys.version_info.major

#mainscript = os.path.join('Priithon', 'PriShell.py')
with open('Priithon/version.py') as h:
    line = h.readline()
    exec(line) # version='XXX'

packages = ['Priithon', 'Priithon.plt']

# -------- Execute -----------------
setup(
    name="Priithon",
    author='Atsushi Matsuda',
    version=version,
    install_requires=['numpy', 'scipy'],
    extras_require={'full': ['wxpython', 'pyopenGL', 'wxmplot']},
    packages=packages,
    entry_points={"gui_scripts": ['Priithon=Priithon.PriShell:main']}
)
