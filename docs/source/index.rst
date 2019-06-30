.. Py_xDH documentation master file, created by
   sphinx-quickstart on Tue Nov 27 16:15:23 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xDH 在 Python 下实现的简易教程
==============================

在这份文档中，我们将会介绍 xDH 型函数的典型：XYG3，其在 Python 下的实现过程。同时，为了方便，这里大量使用 PySCF API 进行中间矩阵的输出与计算。作者希望，读者可以借助于这些成熟量化软件接口，以及 NumPy 对矩阵、张量计算的强大支持，可以较为轻松地在 Post-HF 或 DFT 方法理论推导，与上机实验的结果进行相互比对，并最终享受亲手实现新方法的乐趣。

.. toctree::
   :maxdepth: 2
   :caption: 目录
   :numbered:

   motive
   intro/index
   pyscf/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
