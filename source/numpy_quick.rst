NumPy 快速入门
==============

在这份教程中，我们会大量地使用 NumPy 进行矩阵计算，其使用频率大于我们对于量子化学程序 API 的调用．因此，我们需要对其使用作简单的说明．NumPy 的功能实际上可以很强大，但在这里我们只涉及教程文档中所出现的功能．该节的不少内容可以到 NumPy 的官方 `快速入门 <https://docs.scipy.org/doc/numpy/user/quickstart.html>`_ 中查看．

在继续阅读文档前，请在 Python Consoles 或 Jupyter 中加入 NumPy 环境
::

   >>> import numpy as np

.. attention ::
   如果你已经通过 Python 写过一个 SCF 程序或者其它科学计算程序，那么相信这一节的大部分内容你可以跳过．如果你只使用过 NumPy 的矩阵乘积而没有用过张量乘积，你可以参考 `numpy.einsum <https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html>`_ 文档．

:code:`numpy.ndarray` 对象
--------------------------

:code:`numpy.ndarray` 是 NumPy 最基础的对象，矩阵、张量等都以此储存．它可以由列表构建：
::

   >>> a = np.array(
   ...     [[0, 1, 2],
   ...      [3, 4, 5]])
   >>> a
   array([[0, 1, 2],
          [3, 4, 5]])

其维度可以通过 :code:`ndarray.shape` 给出，维度的储存格式是 tuple 而非 list：
::

   >>> a.shape
   (2, 3)

NumPy 的索引方式与 C、Python 相同，为行索引、存在零元．因此习惯 Fortran 的话可能会不适应：
::

   >>> a[1,0]
   3

上述的示例是二维向量，即矩阵；高维向量，或称张量，可以用相同的方式构建．

矩阵运算
--------

现在定义
::

   >>> b = np.array(
   ...     [[6, 7, 8],
   ...      [9, 10, 11]])

矩阵转置
~~~~~~~~

矩阵的转置可以通过下述三个语句实现：
::

   >>> np.transpose(a)
   >>> a.transpose()
   >>> a.T
   array([[0, 3],
          [1, 4],
          [2, 5]])

对于张量的角标更换，在后续的教程中会使用 :code:`ndarray.swapaxes` 方法或者带传入参数的 :code:`ndarray.transpose`．

矩阵元素运算
~~~~~~~~~~~~

通常的运算符都是元素运算 (elementwise)．这包括向量对数的、向量对向量的、高维向量对低维度向量的运算．
::

   >>> # 指数
   >>> a ** 0.7
   array([[0.        , 1.        , 1.62450479],
          [2.15766928, 2.63901582, 3.08516931]])
   >>> # 加法
   >>> a + b
   array([[ 6,  8, 10],
          [12, 14, 16]])
   >>> # 乘法
   >>> a[0] * b
   array([[ 0,  7, 16],
          [ 0, 10, 22]])
   >>> # 函数
   >>> np.sin(a)
   array([[ 0.        ,  0.84147098,  0.90929743],
          [ 0.14112001, -0.7568025 , -0.95892427]])

.. tip ::
   对于两矩阵之间元素的乘法，会在实际的量化计算中使用到．譬如，若已有原子轨道基组的 :math:`x` 方向偶极积分矩阵 :math:`\boldsymbol{\mu}^x` 与单电子密度矩阵 :math:`\mathbf{P}`，则 :math:`x` 方向偶极矩则为 :math:`\mu^x = \sum_{\mu \nu} \mu_{\mu \nu}^x P_{\mu \nu}`．

除了普通的运算符外，NumPy 支持 :code:`+=` 与 :code:`*=` 等运算符：
::

   >>> # 定义零矩阵
   >>> c = np.zeros(a.shape)
   >>> c
   array([[0., 0., 0.],
          [0., 0., 0.]])
   >>> c += a
   >>> c /= b
   >>> c
   array([[0.        , 0.14285714, 0.25      ],
          [0.33333333, 0.4       , 0.45454545]])

矩阵与张量乘积运算
~~~~~~~~~~~~~~~~~~

对于二元矩阵，矩阵乘积可以用三种方式实现：
::

   >>> np.dot(a, b.T)
   >>> a.dot(b.T)
   >>> a @ b.T
   array([[ 23,  32],
          [ 86, 122]])

对于更高纬度的张量，通常使用 Einstein Convention 的求和记号来写 NumPy 代码．

.. admonition :: Einstein Convention

   若对于二元矩阵乘积 :math:`\mathbf{C} = \mathbf{A} \mathbf{B}`，通常的记号会将上式具象化为

   .. math ::

      C_{ij} = \sum_{k} A_{ik} B_{kj}

   这种记号中，对于 :math:`k` 的求和记号有时会显得冗余，且在排版上显得复杂．Einstein Convention 则略去这种求和．因此，上式可以写作

   .. math ::

      C_{ij} = A_{ik} B_{kj}

   在处理类似于张量乘积譬如双电子电子积分计算、多矩阵相乘譬如原子轨道与分子轨道单电子积分矩阵的转换等情形时，用 Einstein Convention 书写代码会显得非常方便．

普通的矩阵乘积 :math:`C_{ij} = A_{ik} B_{kj}^\mathrm{T}` 可以写作
::

   >>> # 等价于 a.dot(b.T)
   >>> np.einsum('ik, jk -> ij', a, b)
   array([[ 23,  32],
          [ 86, 122]])

普通矩阵乘积的和 :math:`c = A_{ij} B_{ij}` 可以写作
::

   >>> 等价于 (a * b).sum()
   >>> np.einsum('ij, ij ->', a, b)
   423

:code:`numpy.einsum` 效率考量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

尽管矩阵乘积上，:code:`numpy.einsum` 的使用也许是增加工作负担；但相信在实际接触量子化学计算时，会越发地感到使用 :code:`numpy.einsum` 的便利；但该函数通常不是非常效率．为了避免它可能产生的效率问题，这里简单地对该函数作评价．由于该函数现在仍然在改进，因此下述的结论未必在将来成立．

IPython 与 :code:`timeit`
:::::::::::::::::::::::::

在进行下面几个测评前，我们先了解其中两种计算 Python 程序运行时间的的手段：:code:`time` 与 :code:`timeit`．由于在 IPython 下这些评测方式将异常简单，因此这里只介绍 IPython 的用法．由于 Jupyter 基于 IPython，因此也可以使用下面的方法测评；但 Python Consoles 不可．

.. attention ::
   下述的代码由于使用了 IPython 的 `Magic Command <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_，因此只能在 IPython 或 Jupyter 下执行命令，即使下述的代码块使用了传统的 Python Consoles 的风格．

:code:`%time` 将会给出运行一次一行命令时所需要耗费的 CPU 时间 (实际计算时间)、挂墙时间 (Wall time，包含磁盘 I/O、可能产生的其它系统调用、内存资源回收等时间消耗)．对于测算算法效率，可以使用 CPU 时间；而若考察程序的实际运行状况，则应该采用挂墙时间．
::

   >>> %time d = {i for i in range(10000000)}
   CPU times: user 531 ms, sys: 1.23 s, total: 1.77 s
   Wall time: 1.77 s

:code:`%timeit` 将会给出多次运行一行命令时所需要消耗的平均时间．尽管它接近于挂墙时间，但它不考虑 Python 所出现的内存资源回收 (`Garbage Collection <https://docs.python.org/3/glossary.html#term-garbage-collection>`_) 的时间消耗；因此一般来说 :code:`timeit` 所给出的平均时间比起 :code:`time` 所给出的挂墙时间要少一些．不过 :code:`timeit` 命令会尝试多次执行，因此时间会跑得长一些．该命令也是通常评测代码效率所更推荐的方法．
::

   >>> %timeit d = {i for i in range(10000000)}
   1.56 s ± 42.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

如果需要在一个 Cell 而非一行代码中中评测时间消耗，则需要使用 :code:`%%time` 与 :code:`%%timeit` 分别代替 :code:`%time` 与 :code:`%timeit`．

.. note ::
   在 Windows 下，执行 :code:`%time` 后不会出现 CPU 时间．这是作为操作系统的 Windows 所给予的限制．在非 Windows 系统，包括 WSL，则会显示 CPU 时间．

多矩阵连乘
::::::::::

对于矩阵连乘 :math:`R_{im} = r_{ij} r_{jk} r_{kl} r_{lm}`，至少有三种做法；若 :math:`\mathbf{r}` 是由 NumPy 生成的随机 50 维矩阵，则
::

   >>> r = np.random.rand(50, 50)
   >>> %timeit R = r @ r @ r @ r
   26.1 µs ± 1.66 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
   
   >>> %timeit R = np.einsum("ij, jk, kl, lm -> im", r, r, r, r)
   1.72 s ± 6.94 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
   
   >>> %timeit R = np.einsum("ij, jk, kl, lm -> im", r, r, r, r, optimize=True)
   286 µs ± 5.79 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

因此，完成上述命令的最快方式显然是传统的矩阵乘积．对于多矩阵的乘积，:code:`numpy.einsum` 会使用未优化计算复杂度的方式进行计算 (就本例而言，计算复杂度是 :math:`O (N^5)`；但通常我们都会认为上述运算的复杂度在 :math:`O (N^3)` 至 :math:`O (N^2 \log N)` 之间)．而经过优化的 :code:`numpy.einsum` 则可以正确地处理上述计算为不高于 :math:`O (N^3)` 的复杂度，在 50 维下其计算效率比未优化的 :code:`numpy.einsum` 要高效一些，但为此有不小的效率损耗．

不过，如果矩阵维度变小，未优化过的 :code:`numpy.einsum` 反而会快一些．我们现在看看三维矩阵的情况：
::

   >>> r = np.random.rand(3, 3)
   >>> %timeit R = r @ r @ r @ r
   2.5 µs ± 101 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
   
   >>> %timeit R = np.einsum("ij, jk, kl, lm -> im", r, r, r, r)
   11.9 µs ± 655 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
   
   >>> %timeit R = np.einsum("ij, jk, kl, lm -> im", r, r, r, r, optimize=True)
   217 µs ± 2.25 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

因此，论效率上，公式表达式与程序代码关系不友好的矩阵相乘记号是最快的；而使用 :code:`numpy.einsum` 不是最效率的；同时，如果处理的问题维度较小，或不优化与优化的计算复杂度没有改变时，使用未优化的 :code:`numpy.einsum` 有时比优化的版本还快一些．

当然，作为开发方法的工作者，自然会对效率上的要求有所降低，因此，通常情况下直接使用优化的 :code:`numpy.einsum` 未尝不可，因为它的代码本身与公式的对应关系非常显然．很多时候，教程中就会使用这种可能偏低效的方法了．

矩阵构建
--------

创建一个新的全零矩阵可以通过两种途径：
::

   >>> # 通过向 numpy.zeros 传入 tuple 型数组
   >>> np.zeros((2, 3))
   >>> # 也可以通过已有矩阵所导出的 tuple 作为变量
   >>> np.zeros(a.shape)
   >>> # 或者使用 numpy.zeros_like 来构建与传入矩阵相同维度的全零矩阵
   >>> np.zeros_like(a)
   array([[0, 0, 0],
          [0, 0, 0]])

创建对角阵则可以使用
::

   >>> np.eye(3)
   array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]])

而经常地，我们会从本征值向量 :math:`\boldsymbol{e}` 展开成二维分子轨道 Fock 矩阵 :math:`\mathbf{F}`，这个过程通常可以由下述技巧完成：
::

   >>> dim = 4
   >>> e = np.arange(dim)
   >>> e * np.eye(dim)
   array([[0., 0., 0., 0.],
          [0., 1., 0., 0.],
          [0., 0., 2., 0.],
          [0., 0., 0., 3.]])

而在处理 MP2 计算时，其分母项中会出现张量 :math:`\mathcal{E}_{ab}^{ij} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b`；在这里我们以比较简单的矩阵 :math:`\mathcal{E}_{c}^{k} = \varepsilon_k - \varepsilon_c` 来举例子．我们可以通过改变矩阵的维度的技巧获得：
::

   >>> # 定义变量
   >>> dim = 4
   >>> k = np.arange(-1, -dim - 1, -1)
   >>> c = np.arange(2, 2 * dim + 2, 2)
   >>> k
   array([-1, -2, -3, -4])
   >>> c
   array([2, 4, 6, 8])
   >>> # 计算矩阵
   >>> k.reshape(-1, 1)  # 或 k.reshape(4, 1)
   array([[-1],
          [-2],
          [-3],
          [-4]])
   >>> k.reshape(-1, 1) - c  # 即 E_c^k 矩阵
   array([[ -3,  -5,  -7,  -9],
          [ -4,  -6,  -8, -10],
          [ -5,  -7,  -9, -11],
          [ -6,  -8, -10, -12]])

其中用到了矩阵或向量的大小重新定义的函数 :code:`numpy.reshape`．该函数输入为新矩阵大小的 tuple 型变量；也支持用 -1 让程序推断该维度的值：
::

   >>> a.reshape(3, 2)
   >>> a.reshape(-1, 2)
   >>> a.T
   array([[0, 3],
          [1, 4],
          [2, 5]])

如果只是将矩阵压平成为向量，还可以使用 :code:`numpy.ravel` 函数：
::

   >>> a.reshape(-1)
   >>> a.ravel()
   array([0, 1, 2, 3, 4, 5])

向量视图
--------

在这次教程中，出现了少数代码，这些代码的理解必须要基于简单的 NumPy 向量的向量视图 (View) 的概念．这些概念不存在于 Fortran 与 C，它与 Python 本身不具有明确指针多少有些关系．我们知道，Fortran 的向量通常就可以当做指针来看待；而 C 或 C++ 的向量还多一种引用的描述方式．对于 Python，它一般不太容易写出其引用与指针，因此我们不太容易把握在完成向量操作时，是否真的对原来的向量作了操作，导致了原始数据的破坏；或者是否我们复制出一个新的向量，造成了内存空间的浪费．

NumPy 的向量类可以简单地看作由底层数据和表面形状 (shape) 构成．NumPy 很少采用真正的深层复制 (Deep Copy)，即很少将底层数据复制到另一个变量中．深层复制的通常做法是
::

   >>> d = a.copy()

以后对 :code:`d` 的任何数据、形状的改动，都不会影响 :code:`a`．反之亦然．

而更多时候是引用．它不将数据复制出来，但包含表面形状的信息．在最为简单的情况下，可以直接理解为一种引用．例如，向量的索引相当于对其对应的原始数据的引用：
::

   >>> d = np.arange(4)
   >>> d[2] = 10
   >>> d
   array([ 0,  1, 10,  3])

但还有一些更为特殊的操作，这些不能简单地看作通常的引用．例如我们可以令 :code:`v` 是 :code:`d` 的一种视窗：:code:`v` 是 :code:`d` 若干个元素的引用；对 :code:`v` 的形状的改变不会对 :code:`d` 产生影响，但对其数据的改动则会直接改动 :code:`d` 的数据：
::

   >>> d = np.arange(4)
   >>> # v 是 d 的视图，并非将数据复制给了 v，数据还是从 d 读出来
   >>> v = d[0:3:2]
   >>> v
   array([0, 2])
   >>> # 更改 v 的形状对 d 没有影响
   >>> v.shape = 2, 1
   >>> v
   array([[0],
          [2]])
   >>> d.shape
   (4,)
   >>> # 更改 v 的数据对 d 有影响，这类似于引用关系
   >>> v[:] = np.array([[-2], [-6]])
   >>> d
   array([-2,  1, -6,  3])
   >>> # 但下面这句语句并非是给视图更改数据
   >>> # 创建了新的向量赋值给 v，自此 v 与 d 不存在相互关系
   >>> v = np.array([[-3], [-9]])
   >>> d
   array([-2,  1, -6,  3])

其它函数
--------

:code:`numpy.linalg.norm` 模长函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于向量模长，可以简单地调用它来计算；对于矩阵，它等同于化为向量：
::

   >>> np.linalg.norm(a)
   >>> np.linalg.norm(a.ravel())
   7.416198487095663

:code:`numpy.linalg.eigh` 本征系统
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若现在有对称矩阵 :math:`\textbf{f}`，则其本征值与本征向量可以借助该函数获得：
::

   >>> f = np.array(
   ...     [[2., 3., 3.],
   ...      [3., 2.33, 3.],
   ...      [3., 3., 3.]])
   >>> eig, vec = np.linalg.eigh(f)
   >>> # 本征值
   >>> eig
   array([-0.85515066, -0.27773769,  8.46288834])
   >>> # 本征向量
   >>> vec
   array([[-0.7806939 , -0.29943621, -0.54850251],
          [ 0.61076205, -0.55134403, -0.56832163],
          [ 0.13223751,  0.77868974, -0.61331519]])

:code:`numpy.allclose` 判断矩阵相同
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

现在我们对本征方才的本征系统的简单性质作验证．首先，本征向量所构成的矩阵是正交矩阵，即其逆应当等于转置：
::

   >>> vec.T
   array([[-0.7806939 ,  0.61076205,  0.13223751],
          [-0.29943621, -0.55134403,  0.77868974],
          [-0.54850251, -0.56832163, -0.61331519]])
   >>> np.linalg.inv(vec)
   array([[-0.7806939 ,  0.61076205,  0.13223751],
          [-0.29943621, -0.55134403,  0.77868974],
          [-0.54850251, -0.56832163, -0.61331519]])
   >>> np.allclose(vec.T, np.linalg.inv(vec))
   True

本征系统本质上可以看作是一种矩阵对角化．我们验证一下对角化前后的矩阵是否一致：
::

   >>> f_after_diag = vec @ (eig * np.eye(eig.shape[0])) @ vec.T
   >>> f_after_diag
   array([[2.  , 3.  , 3.  ],
          [3.  , 2.33, 3.  ],
          [3.  , 3.  , 3.  ]])
   >>> np.allclose(f_after_diag, f)
   True
