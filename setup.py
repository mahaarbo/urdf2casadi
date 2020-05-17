#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "urdf2casadi"
DESCRIPTION = "Module for turning a chain in a URDF to a casadi function."
URL = "http://github.com/mahaarbo/urdf2casadi"
EMAIL = "mathias.arbo@ntnu.no"
AUTHOR = "Mathias Hauan Arbo"
REQUIRES_PYTHON = ">=2.7.0"
VERSION = "0.1.0"

REQUIRED = [
    "casadi", "numpy", "urdf_parser_py"
]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    # long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    # If your package is a single module, use this instead of ""packages"":
    # py_modules=[""mypackage""],

    # entry_points={
    #     ""console_scripts"": [""mycli=mymodule:cli""],
    # },
    install_requires=REQUIRED,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Topic :: Scientific/Engineering :: Mathematics"
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5"
        "Programming Language :: Python :: 3.6"
        "Programming Language :: Python :: Implementation :: CPython"
        "Programming Language :: Python :: Implementation :: PyPy"
    ],

)
