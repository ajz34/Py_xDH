# Python xDH Project

[![Build Status](https://travis-ci.com/ajz34/Py_xDH.svg?branch=master)](https://travis-ci.com/ajz34/Py_xDH)
[![codecov](https://codecov.io/gh/ajz34/Py_xDH/branch/master/graph/badge.svg)](https://codecov.io/gh/ajz34/Py_xDH)
[![Documentation Status](https://readthedocs.org/projects/py-xdh/badge/?version=latest)](https://py-xdh.readthedocs.io/zh_CN/latest/?badge=latest)

This project is mainly documentation or notes of some basic quantum chemistry derivative implementations
(including GGA, MP2, non-consistent GGA, xDH).
**Documentation is written in Chinese for the current moment.**

This project also includes demo python package `pyxdh` that implements properties calculation, which integral engine,
basis processing, DFT grid engine and CP-HF algorithm is based on python quantum chemistry package [PySCF](https://github.com/pyscf/pyscf).

这项工程主要是一些基础的量化方法 (涵盖 GGA 为基础的自洽场、MP2、非自洽 GGA、XYG3 型双杂化泛函) 的梯度性质计算的说明文档与笔记。
**目前这份笔记仅有中文的版本。**

这项工程也包含实现这些梯度性质计算的 Python 库 `pyxdh`。该库的电子积分、基组处理、DFT 格点积分引擎与 CP-HF 方程算法基于
[PySCF](https://github.com/pyscf/pyscf) 的量子化学库。

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
  
其中，
  - "+"：在这份代码库中已经实现；
  - "-"：在另一份私有代码库中已经实现；
  - "GGA"：以 GGA 为基础的 SCF，包括 HF；但 LDA、meta-GGA、NLC 现不支持；
  - "MP2"：以 SCF 为参考态的二阶微扰；这包括普通的 MP2 和 B2PLYP 型双杂化泛函；
  - "GGA-GGA"：非自洽 GGA，譬如以 HF 为参考态获得的 B3LYP 能量的泛函；
  - "GGA xDH"：以 GGA 为参考态与包含 GGA 的能量泛函的 XYG3 型双杂化泛函。

## Documentation

**Note: Documentation remains to be updated! Only `pyxdh` code can be used currently.**

**注意：文档部分仍有待完善！目前为止，`pyxdh` 库的代码确实是可以使用的，但一些文档的代码是有问题的。**

Published web page: https://py-xdh.readthedocs.io/zh_CN/latest/

Prerequisite knowledge of chapter 3, 4, 10 of *A New Dimension to Quantum Chemistry: Analytic Derivative Methods in
Ab Initio Molecular Electronic Structure Theory*, Yamaguchi, *et. al.* or equivalent is recommended; while chapter
1, 2, 3, 6 of *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Szabo and Ostlund or
equivalent is essential.

阅读文档建议对于 *A New Dimension to Quantum Chemistry: Analytic Derivative Methods in Ab Initio Molecular Electronic
Structure Theory*, Yamaguchi, *et. al.* 书中第 3、4、10 章节或相似的内容有所了解；若不了解，*Modern Quantum Chemistry:
Introduction to Advanced Electronic Structure Theory*, Szabo and Ostlund 第 1、2、3、6 章或者等同的内容的了解是必须的。

Documentation consists of:

  - Basic and specific usage of Python, NumPy;
  - Matrices and derivative matrices realization and derivation;
  - Properties realization;
  - Plenty codes with abundant fomular explanation.

Documentation consists of executable Jupyter notebooks. In order to run these Jupyter notebooks, one may pytest `pyxdh`
package in terminal successfully.

文档包含以下内容：

  - Python、NumPy 部分基础，以及与文档和工程有关的用法介绍；
  - 量化矩阵及其导数矩阵的实现与推导；
  - 分子性质的实现；
  - 充足的代码与公式解释；

文档主要由可执行的 Jupyter 笔记本构成。若要执行这些 Jupyter 笔记本，请先确保对 `pyxdh` 库的 pytest 是成功的。

## `pyxdh` Package

This is merely a demo package that implements derivative of some basic quantum chemistry methods.
Currently it is not distrubuted to PyPI.

这个库目前只是一个短小的、包含基础量化方法梯度实现的库。它并没有发布到 PyPI 中。

### Deficiencies and facilities

Deficiencies can be:

  - Numerical behavior in big molecule is not tested;
  - Huge time cost when evaluating B2PLYP-type functional hessian;
  - Huge memory cost O(N^4) for MP2 properties, as well as no density fitting is utilized;
  - Complicated multiple inheritance (diamond inheritance);
  - Does not support all kind of DFT approximations;
  - Code strongly disagree with "pure function" philosophy;
  - The author believe code from a junior chemistry student major in chemistry should not be believed in any way;
    this kind of code is somehow like homework projects of advanced PhD courses.

这个库目前的缺陷有

  - 缺少对于大分子的数值测评；
  - 对于 B2PLYP 型泛函，Hessian 计算的时间消耗过于严重；
  - 所有 MP2 方法不使用 Density Fitting，并且会有 O(N^4) 的内存消耗；
  - 使用了多重继承 (菱形继承)；
  - 不支持所有 DFT 近似的计算；
  - 与 "pure function" 的思想背道而驰；
  - 作者认为不可以信任一个修读化学的低年级学生的代码；并且这类代码相比与成熟的库，更像是高级 PhD 课程的大作业。

However, `pyxdh` code is intended to be:

  - Easy to use, since no sophiscated compilation is required;
  - Easy to extend and contribute with object-oriented designed code framework;
  - Intuitive equation to code transformation and vice versa;
  - Code quality (coverage) guaranteed.

然而，`pyxdh` 的代码希望是

  - 由于没有复杂的编译过程，应当代码易于使用；
  - 由于使用面向对象的思想，程序应当易于扩展；
  - 具有直观的公式与代码互推；
  - 可观的代码覆盖率。

The author hope this package, with its documentations, can be good education or exercise material to
theoretical/computational chemistry/physics PhD students in his/her first year;
but not the program ability (derivative properties calculation) itself.

作者希望这个库与其文档可以是一份对于理论/计算化学/物理 PhD 学生不错的学习与练习材料，而并不是希望大家关注这个程序的功能本身。

### Usage

- Copy `.pyscf_conf.py` to `$HOME` to increase CP-HF precision.
  It can be crucial when evaluating molecular coefficient derivative matrix precisely.
- `export PYTHONPATH=$Py_xDH/pyxdh:$PYTHONPATH`; or install package with pip manually.
  Latter approach has not been tested. IDE like PyCharm is recommended when working with python codes.
- Testing classes in source code can be examples for running jobs. Hacking these code is appreciated.

Generally, following instructions in `.travis.yml` is okay.
All tests may cost 5-20 minutes depending on computers or servers. 

- 请先复制 `.pyscf_conf.py` 文件到 `$HOME` 文件夹；这通常会提高 CP-HF 方程精度，并因此会对矩阵梯度的正确性有至关重要的影响。
- 请执行 `export PYTHONPATH=$Py_xDH/pyxdh:$PYTHONPATH`，或者直接安装该库；但后者没有经过测试。请尽量使用类似于 PyCharm 等集成开发环境来执行程序代码。
- 代码中的测试样例也可以是代码的执行样例。这些代码可以作为参考。

一般来说，按照 `.travis.yml` 文件的指示来运行程序也是可以的。一般来说，根据电脑或服务器的情况不同，运行所有测试需要 5-20 分钟。

## Acknowledge

- [PySCF](https://github.com/pyscf/pyscf) inspirits this project a lot!
- [Psi4NumPy](https://github.com/psi4/psi4numpy) is the initial motivation for this project. However, for some practical
  reasons, this project has been moved to PySCF.
- Thanks labmates for valuable discussions and suggestions.
- Thanks supervisor and teachers in lab for project support and server support.
- Thanks parents for project support.
- Currently, the author does not know any funding grants supporting this project. It should have some.
  Project is almost personally driven at this stage.
- Futher discussion is welcomed by raising issue or e-mail. Chinese is prefered; English is also okay.
