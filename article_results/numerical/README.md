# Numerical testing
## Requirements
These examples are created for numerically testing `urdf2casadi` against KDL, RBDL and pybullet.

Install `kdl_parser_py` by running `sudo apt-get install ros-VERSION-kdl-parser ros-VERSION-kdl-parser-py`. 

Install RBDL from [source](https://github.com/rbdl/rbdl). Remember to set `RBDL_BUILD_ADDON_URDFREADER`, `RBDL_BUILD_PYTHON_WRAPPER`. You may have to change `PYTHON_INCLUDE_DIR` to `/usr/include/python2.7;/usr/local/lib/python2.7/dist-packages/numpy/core/include`.

Install pybullet by running `sudo pip install pybullet`.
