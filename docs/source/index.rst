.. Py_xDH documentation master file, created by
   sphinx-quickstart on Tue Nov 27 16:15:23 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xDH 在 Python 下实现的简易教程
==============================

在这份文档中，我们将会介绍 xDH 型函数的典型：XYG3，其在 Python 下的实现过程。同时，为了方便，这里大量使用 PySCF API 进行中间矩阵的输出与计算。作者希望，读者可以借助于这些成熟量化软件接口，以及 NumPy 对矩阵、张量计算的强大支持，可以较为轻松地在 Post-HF 或 DFT 方法理论推导，与上机实验的结果进行相互比对，并最终享受亲手实现新方法的乐趣。

.. toctree::
   :caption: 动机

   motive

.. toctree::
   :caption: 前置准备
   :numbered:

   intro/index
   intro/build_environment
   intro/intro_python
   intro/intro_numpy
   intro/adv_numpy
   intro/intro_pyxdh

.. toctree::
   :caption: 量化基础必要背景
   :numbered:

   qcbasic/index
   qcbasic/basis_integral
   qcbasic/basic_rhf
   qcbasic/basic_mp2
   qcbasic/basic_grid
   qcbasic/basic_lda
   qcbasic/basic_gga
   qcbasic/basic_ncgga
   qcbasic/basic_b2plyp
   qcbasic/basic_xyg3
   qcbasic/proj_xyg3

.. toctree::
   :caption: 数值梯度必要背景
   :numbered:

   numdiff/index
   numdiff/basic_num
   numdiff/nuc_grad
   numdiff/num_dip
   numdiff/num_hess
   numdiff/num_polar
   numdiff/num_dipderiv

.. toctree::
   :caption: 一阶梯度与性质
   :numbered:

   derivonce/index
   derivonce/ref_list
   derivonce/skeleton_and_U
   derivonce/dip_rhf_skeleton
   derivonce/grad_rhf_skeleton
   derivonce/grad_rhf_U
   derivonce/grad_mp2
   derivonce/grad_gga
   derivonce/grad_bdh
   derivonce/grad_xdh
   derivonce/response_density
   derivonce/dip_xdh

.. toctree::
   :caption: 二阶梯度与性质
   :numbered:

   derivtwice/index
   derivtwice/hess_rhf
   derivtwice/hess_rhf_U
   derivtwice/hess_mp2_unsafe
   derivtwice/hess_mp2_safe
   derivtwice/hess_gga
   derivtwice/hess_gga_U

.. toctree::
   :caption: 开壳层开发笔记
   :numbered:

   unrestricted/uks_details


索引
====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
