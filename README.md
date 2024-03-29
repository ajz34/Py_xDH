# Python xDH Project

**This project is under going code refactorization. Document code is currently not updated. Unrestricted and RI implementation is on-going. Also refers to unfinished [dh project](https://github.com/ajz34/pyscf-forge/tree/pre-0.2).**

**程序正在进行重构。其所对应的文档目前还没有更新。开壳层与 RI 实现目前还有待完善。同时参见尚未完成的 [dh 项目](https://github.com/ajz34/pyscf-forge/tree/pre-0.2)。**

|         | Badges   |
| :------ | :------- |
| **Builds** | [![Build Status](https://travis-ci.com/ajz34/Py_xDH.svg?branch=master)](https://travis-ci.com/ajz34/Py_xDH) [![Documentation Status](https://readthedocs.org/projects/py-xdh/badge/?version=latest)](https://py-xdh.readthedocs.io/zh_CN/latest/?badge=latest) ![GitHub](https://img.shields.io/github/license/ajz34/py_xdh) |
| **Code Quality** | [![codecov](https://codecov.io/gh/ajz34/Py_xDH/branch/master/graph/badge.svg)](https://codecov.io/gh/ajz34/Py_xDH) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ajz34/Py_xDH.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ajz34/Py_xDH/context:python) |
| **Docker** | ![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/ajz34/pyxdh) [![](https://images.microbadger.com/badges/image/ajz34/pyxdh.svg)](https://microbadger.com/images/ajz34/pyxdh "Get your own image badge on microbadger.com") |

This project is mainly documentation or notes of some basic quantum chemistry derivative implementations
(including GGA, MP2, non-consistent GGA, xDH).
**Documentation is written in Chinese for the current moment.**

This project also includes demo python package `pyxdh` that implements properties calculation, which integral engine,
basis processing, DFT grid engine and CP-HF algorithm is based on python quantum chemistry package [PySCF](https://github.com/pyscf/pyscf).

这项工程主要是一些基础的量化方法 (涵盖 GGA 为基础的自洽场、MP2、非自洽 GGA、XYG3 型双杂化泛函) 的梯度性质计算的说明文档与笔记。
**目前这份笔记仅有中文的版本。**

这项工程也包含实现这些梯度性质计算的 Python 库 `pyxdh`。该库的电子积分、基组处理、DFT 格点积分引擎与 CP-HF 方程算法基于
[PySCF](https://github.com/pyscf/pyscf) 的量子化学库。

> **Warning**
> 
> `pyxdh` has not been fully and rigorously tested, nor peer reviewed.
> Please use other quantum chemistry software instead anyway if one is preparing papers or making industrial implementation.
> 
> This package is virtually only for learning coding techniques for double hybrid secondary derivative properties.
> Efficiency could be extremely terrible and is not the gist of this package.
> 
> `pyxdh` 没有经过严格的测评，目前也没有任何同行评议。
> 在这份警告撤销之前，请不要在正式发表的论文中使用此处的做法作为 XYG3 及其导数性质的计算方法。
> 对于其它方法，譬如 MP2、双杂化泛函等性质，也请在生产环境或正式发表的论文中使用成熟的量化软件。
> 
> 这个库仅仅是对二阶梯度初步实现的技术讨论。程序效率会比较糟糕且并不是这个库关心的核心问题。

## Version Information

Current version of pyxdh is 0.0.7. This version should work with pyscf==1.7.5.

Previous version 0.0.3 should work with pyscf==1.6.4.

## Abilities

|                 | HF        | GGA  | MP2       | GGA-GGA | GGA xDH |
|:---------------:|:----------|:-----|:----------|:--------|:--------|
| Energy          | R, U, RDF | R, U | R, U, RDF | R, U    | R, U    |
| Gradient        | R, U, RDF | R, U | R, U      | R, U    | R, U    |
| Dipole          | R, U      | R    | R, U      | R, U    | R, U    |
| Hessian         | R, U      | R    | R         | R       | R       |
| Dipole Gradient | R, U      | R    | R, U      | R       | R       |
| Polarizability  | R, U      | R    | R, U      | R       | R       |

Where
  - "R", "U", "RDF", "UDF": Restricted, Unrestricted, Restricted Density Fitting, Unrestricted Density Fitting;
  - "GGA": SCF process with GGA kernel or naive HF; note that LDA, meta-GGA or NLC is not supported in these code;
  - "MP2": PT2 with SCF reference; can be naive MP2 with HF reference or B2PLYP-type Double Hybrid functional (DH);
  - "GGA-GGA": Non-Consistent GGA, e.g. B3LYP energy take HF density as reference;
  - "GGA xDH": XYG3 type functional (xDH) take GGA density as reference.
  
其中，
  - "R", "U", "RDF", "UDF": 闭壳层、开壳层、闭壳层 Density Fitting、开壳层 Density Fitting；
  - "GGA"：以 GGA 为基础的 SCF，包括 HF；但 LDA、meta-GGA、NLC 现不支持；
  - "MP2"：以 SCF 为参考态的二阶微扰；这包括普通的 MP2 和 B2PLYP 型双杂化泛函；
  - "GGA-GGA"：非自洽 GGA，譬如以 HF 为参考态获得的 B3LYP 能量的泛函；
  - "GGA xDH"：以 GGA 为参考态与包含 GGA 的能量泛函的 XYG3 型双杂化泛函。

## Example: Calculate XYG3 Polarizability

Example is explained in [Documentation](https://py-xdh.readthedocs.io/zh_CN/latest/intro/intro_pyxdh.html)
or in [jupyter page](https://github.com/ajz34/Py_xDH/blob/master/docs/source/intro/intro_pyxdh.ipynb).

Following is the almost the same code demo extracted from these documents.

```python
from pyxdh.DerivOnce import DipoleXDH
from pyxdh.DerivTwice import PolarXDH
from pyscf import gto, dft

# Generate H2O2 molecule
mol = gto.Mole()
mol.atom = """
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
"""
mol.basis = "6-31G"
mol.verbose = 0
mol.build()

# Generate (99, 590) grids
grids = dft.Grids(mol)
grids.atom_grid = (99, 590)
grids.build()

# Self-consistent part of XYG3
scf_eng = dft.RKS(mol)
scf_eng.xc = "B3LYPg"
scf_eng.grids = grids

# Non-self-consistent GGA part of XYG3
nc_eng = dft.RKS(mol)
nc_eng.xc = "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"
nc_eng.grids = grids

# Dipole helper from pyxdh
config = {
    "scf_eng": scf_eng,
    "nc_eng": nc_eng,
    "cc": 0.3211
}
dip_xDH = DipoleXDH(config)

# Polar helper from pyxdh, generated by dipole helper
config = {
    "deriv_A": dip_xDH,
    "deriv_B": dip_xDH,
}
polar_xDH = PolarXDH(config)

# Final result of polarizability
print(- polar_xDH.E_2)

# Results should be something like
# [[ 6.87997982 -0.1021484  -1.09976624]
#  [-0.1021484   4.7171979   0.29678172]
#  [-1.09976624  0.29678172 14.75690205]]
```

## Documentation

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

This is merely a demo package that implements derivative of some basic quantum chemistry methods. Not for real-world problem computation, and not efficient at all.

这个库目前只是一个短小的、包含基础量化方法梯度实现的库。它不适合用于实际生产环境的计算；它的效率及其糟糕。

### Installation

Currently, PyPI installation is available. Python version 3.8 should work.

```bash
pip install pyxdh
```

See also [docker image](#Docker-Image).

### Deficiencies and facilities

Deficiencies can be:

  - Support restricted reference, no-forzen-core, no-density-fitting methods currently; no symmetry is utilized;
  - Numerical behavior in big molecule is not tested;
  - Huge time cost when evaluating B2PLYP-type functional hessian;
  - Huge memory cost O(N^4) for MP2 properties, as well as no density fitting is utilized;
  - Complicated multiple inheritance (diamond inheritance);
  - Does not support all kind of DFT approximations (especially LDA, meta-GGA, NLC);
  - Code strongly disagree with "pure function" philosophy;
  - The author believe code from a junior chemistry student major in chemistry should not be believed in any way;
    this kind of code is somehow like homework projects of advanced PhD courses.

这个库目前的缺陷有

  - 现在只支持闭壳层参考态，并且不支持冻核近似，以及 Density Fitting 方法；不使用对称性质进行计算上的简化；
  - 缺少对于大分子的数值测评；
  - 对于 B2PLYP 型泛函，Hessian 计算的时间消耗过于严重；
  - 所有 MP2 方法不使用 Density Fitting，并且会有 O(N^4) 的内存消耗；
  - 使用了多重继承 (菱形继承)；
  - 不支持所有 DFT 近似的计算 (譬如 LDA、meta-GGA、NLC)；
  - 与 "pure function" 的思想背道而驰；
  - 作者认为不可以指望一个修读化学的低年级学生的代码；并且这类代码相比与成熟的库，这更像是高级 PhD 课程的大作业。

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
- Python package dependencies is listed in `requirements.txt`. Simply `pip install -r requirements.txt` should work.
- `pyscf` package must be installed from PyPI currently, since `xcfun` dependency does not occur in conda pyscf channel.
- Testing classes in source code can be examples for running jobs. Hacking these code is appreciated.

Generally, following instructions in `.travis.yml` is okay.
All tests may cost 5-20 minutes depending on computers or servers. 

- 请先复制 `.pyscf_conf.py` 文件到 `$HOME` 文件夹；这通常会提高 CP-HF 方程精度，并因此会对矩阵梯度的正确性有至关重要的影响。
- 请执行 `export PYTHONPATH=$Py_xDH:$PYTHONPATH`，或者直接安装该库；但后者没有经过测试。请尽量使用类似于 PyCharm 等集成开发环境来执行程序代码。
- Python 库函数依赖关系列举在 `requirements.txt` 中。执行 `pip install -r requirements.txt` 应当可以安装这部分依赖关系。
- 由于 `xcfun` 包不在 conda 的 pyscf channel，因此现在 `pyscf` 必须通过 PyPI 安装。
- 代码中的测试样例也可以是代码的执行样例。这些代码可以作为参考。

一般来说，按照 `.travis.yml` 文件的指示来安装与运行程序也是可以的。一般来说，根据电脑或服务器的情况不同，运行所有测试需要 5-20 分钟。

### Docker Image

Docker provides another way to use pyxdh package in any system (Linux, Windows, Mac, ...), as well as running documents in jupyter environment.

To use docker image, one can run the following code when connected to internet (provided that 8888 port is available):

Docker 提供了另一种可以在 Linux、Windows、Mac 使用 pyxdh 库的方式；它同时还允许我们直接使用 jupyter 笔记本环境。

若要用 Docker 运行 pyxdh，请在保证网路连通、端口 8888 没有被占用的情况下执行下述代码：

```bash
$ docker run -it -p 8888:8888 ajz34/pyxdh
```

If above code is successful, jupyter notebook should have been run in docker container.
Simply follow the instructions from terminal should work. 

如果上述代码运行顺利，jupyter 笔记本环境应当已经在 Docker 容器里配置完毕并运行了。按照终端所给出的指示打开 Jupyter 笔记本就可以了。

## Acknowledge

- [PySCF](https://github.com/pyscf/pyscf) inspirits this project a lot!
- [Psi4NumPy](https://github.com/psi4/psi4numpy) is the initial motivation for this project. However, for some practical
  reasons, this project has been moved to PySCF.
- Thanks labmates for valuable discussions and suggestions.
- Thanks supervisor and teachers in lab for project support and server support.
- Thanks parents for project support.
- Futher discussion is welcomed by raising issue or e-mail. Chinese is prefered; English is also okay.
- Funding information: National Natural Science Foundation of China (Grant 21688102), the Science Challenge Project (Grant TZ2018004), and the National Key Research and Development Program of China (Grant 2018YFA0208600).

This project is not going to be formally published, as it is more like documentation demo instead of program. This project is closely related to the following article:
> Gu, Y.; Zhu, Z.; Xu, X. Second-Order Analytic Derivatives for XYG3 Type of Doubly Hybrid Density Functionals: Theory, Implementation, and Application to Harmonic and Anharmonic Vibrational Frequency Calculations. J. Chem. Theory Comput. 2021, 17 (8), 4860–4871. https://doi.org/10.1021/acs.jctc.1c00457.

