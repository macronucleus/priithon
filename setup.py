"""
Usage:
    python setup.py install

The code was tested with Miniconda verion of python
"""
import os #, sys, stat
from setuptools import setup

dname = os.path.basename(os.path.dirname(__file__))

with open('imgio/version.py') as h:
    line = h.readline()
    exec(line) 

packages = ['imgio', 'imgio.mybioformats']

# -------- Execute -----------------
setup(
    name="imgio",
    author='Atsushi Matsuda',
    version=version,
    install_requires=['numpy'],
    extras_require={'common': ['pillow', 'tifffile', 'nd2', 'czifile', 'oiffile', 'readlif'],
                        'full': ['pillow', 'tifffile', 'nd2', 'czifile', 'oiffile', 'readlif', 'javabridge', 'python-bioformats', 'lxml']},
    packages=packages
)
