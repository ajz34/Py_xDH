# Python xDH Project

[![Build Status](https://travis-ci.com/ajz34/Py_xDH.svg?branch=master)](https://travis-ci.com/ajz34/Py_xDH)
[![codecov](https://codecov.io/gh/ajz34/Py_xDH/branch/master/graph/badge.svg)](https://codecov.io/gh/ajz34/Py_xDH)
[![Documentation Status](https://readthedocs.org/projects/py-xdh/badge/?version=latest)](https://py-xdh.readthedocs.io/zh_CN/latest/?badge=latest)

This project is mainly documentation or notes of some basic quantum chemistry derivative implementations.
Documentation is written in Chinese for the current moment.

This project also includes demo python package that implements the following properties calculation:

## Abilities

|                 | GGA | MP2 | GGA-GGA | GGA xDH |
|:---------------:|:---:|:---:|:-------:|:-------:|
| Energy          | +   | +   | +       | +       |
| Gradient        | +   | +   | +       | +       |
| Dipole          | +   | +   | +       | +       |
| Hessian         | +   | +   | -       | -       |
| Dipole Gradient |     |     |         |         |
| Polarizability  | +   | +   | -       | -       |

Where
  - "+": Implemented in this repository;
  - "-": Implemented in a private repository;
  - "GGA": SCF process with GGA kernel or naive HF; note that LDA, meta-GGA or NLC is not supported in these code;
  - "MP2": PT2 with SCF reference; can be naive MP2 with HF reference or B2PLYP-type Double Hybrid functional (DH);
  - "GGA-GGA": Non-Consistent GGA, e.g. B3LYP energy take HF density as reference;
  - "GGA xDH": XYG3 type functional (xDH) take GGA density as reference.

## Documentation

Documentation is the main purpose of Python xDH project. It is currently written in Chinese.

Published web page: https://py-xdh.readthedocs.io/zh_CN/latest/

Prerequisite knowledge of chapter 3, 4, 10 of *A New Dimension to Quantum Chemistry: Analytic Derivative Methods in
Ab Initio Molecular Electronic Structure Theory*, Yamaguchi, *et. al.* or equivalent is recommended; while chapter
1, 2, 3, 6 of *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Szabo and Ostlund or
equivalent is essential.

Documentation consists of:

  - Basic and specific usage of python, numpy;
  - Matrices and derivative matrices realization and derivation;
  - Properties realization;
  - Plenty codes with abundant fomular explanation.

Documentation consists of executable Jupyter notebooks. In order to run these Jupyter notebooks, one may pytest `pyxdh`
package in terminal successfully.

## `pyxdh` Package

This is merely a demo package that implements derivative of some basic quantum chemistry methods.
Currently it is not distrubuted to pip.

### Deficiencies and facilities

Deficiencies can be:

  - Numerical behavior in big molecule is not tested;
  - Huge time cost when evaluating B2PLYP-type functional hessian;
  - Huge memory cost (O(N^4)) for MP2 secondary derivative properties;
  - Complicated multiple inheritance (diamond inheritance);
  - Does not support all kind of DFT approximations;
  - Code strongly disagree with "pure function" philosophy;
  - The author believe code from a junior chemistry student major in chemistry should not be believed in any way;
    this kind of code is somehow like homework projects of advanced PhD courses.

However, `pyxdh` code is intended to be:

  - Easy to use, since no sophiscated compilation is required;
  - Easy to extend and contribute with object-oriented designed code framework;
  - Intuitive equation to code transformation and vice versa;
  - Code quality (coverage) guaranteed.

The author hope this package, with its documentations, can be good education or exercise material to
theoretical/computational chemistry/physics PhD students in his/her first year;
but not the program ability (derivative properties calculation) itself.

### Usage

- Copy `.pyscf_conf.py` to `$HOME` to increase CP-HF precision.
  It can be crucial when evaluating molecular coefficient derivative matrix precisely.
- `export PYTHONPATH=$Py_xDH/pyxdh:$PYTHONPATH`; or install package with pip manually.
  Latter approach has not been tested. IDE like PyCharm is recommended when working with python codes.
- Testing classes in source code can be examples for running jobs. Hacking these code is appreciated.

Generally, following instructions in `.travis.yml` is okay.
All tests may cost 5-20 minutes depending on computers or servers. 
