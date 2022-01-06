#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "urdf2casadi"
DESCRIPTION = "Module for finding casadi functions of forward kinematics and dynamics of a URDF chain."
URL = "http://github.com/mahaarbo/urdf2casadi"
EMAIL = "mathiasarbo@gmail.com"
AUTHOR = "Mathias Hauan Arbo"
REQUIRES_PYTHON = ">=2.7.0"
VERSION = "1.0.0"

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
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    long_description_content_type="text/markdown",
    install_requires=REQUIRED,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    python_requires=REQUIRES_PYTHON

)
