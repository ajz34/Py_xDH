环境搭建
========

Anaconda 环境
-------------

这份教程的所有文档、计算都通过 Python 实现；因此需要安装一个 Python 环境．`Anaconda <https://www.anaconda.com/>`_ 作为 Python 的一种非官方发行版，它集成了众多科学计算中所必须与经常使用的库；它至少包含可以实现矩阵计算的 NumPy、绘图实现 MatPlotLib、交互笔记本 Jupyter、仿 Matlab IDE 的 Spyder 等库与工具，以及文档工具 Pandoc、Python 管理工具 conda 和 pip 等管理工具．Anaconda 在普遍的科学计算领域足够完备，同时库的依赖关系可以通过 pip 与 conda 等方便地管理，使得程序员不必耗费许多精力在准备环境上．因此它也是普遍推荐的 Python 安装的解决方案．

由于 Psi4 不具备 Windows 版本，因此这里介绍 Linux 下的安装．

.. attention ::
    WSL (Windows Subsystem of Linux) 尽管也是良好的 Linux 环境，但由于 Windows 系统不区分文件名中的大小写，因此在 WSL 下 Anaconda 时可能会遇到问题．因此，推荐在双系统或者服务器上安装 Anaconda．

#. 前往 `清华开源镜像 <https://mirror.tuna.tsinghua.edu.cn/anaconda/archive/>`_ 下，找到最新版本的 Anacodna 的 Linux 版本并下载；若在服务器上不太容易打开网页，则可以使用下述命令行下载 5.3 版本的 Anaconda：
   
   .. code-block :: bash

      $ wget https://mirror.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.0-Linux-x86_64.sh

#. 以 :code:`Anaconda3-5.3.0-Linux-x86_64.sh` 文件为例，在下载文件夹下直接执行：
   
   .. code-block :: bash

      $ chmod +x Anaconda3-5.3.0-Linux-x86_64.sh
      $ ./Anaconda3-5.3.0-Linux-x86_64.sh

   随后按照指示进行安装即可．

   .. tip ::
      安装后，需要前往 :code:`$HOME/.bashrc` 检查是否将该版本的 Anaconda 加入了环境变量；参考安装日志中出现下述输出的上下文：
      ::

         You may wish to edit your $HOME/.bashrc to setup Anaconda3:

      对于服务器用户，可能需要改动的文件不是 :code:`$HOME/.bashrc` 而是 :code:`$HOME/.bash_profile`．

#. 若在中国，可以使用清华镜像的仓库加速 Anaconda 的库的下载速度与更新速度；其使用方式是 (引用自 `Anaconda 镜像使用帮助 <https://mirror.tuna.tsinghua.edu.cn/help/anaconda/>`_)
   
   .. code-block :: bash

      $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
      $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
      $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
      $ conda config --set show_channel_urls yes

#. 除了 conda 外，不少 Python 库在 `PyPI <https://pypi.org/>`_ 库索引上．我们仍然可以使用清华镜像的仓库加速下载 (引用自 `PyPI 镜像使用帮助 <https://mirror.tuna.tsinghua.edu.cn/help/pypi/>`_)

   .. code-block :: bash

      $ pip install pip -U
      $ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#. 若要使用 Anaconda 发行版的 Python，退出当前 Terminal 再重新进入即可．


PySCF 环境搭建
--------------

在这份教程中，我们不需要改动 PySCF 的程序，也不需要进行编译，只需要获得其 Python 版本的二进制可执行文件即可．PySCF 的安装非常简单，只要执行下述语句即可：

   .. code-block :: bash

      $ pip install pyscf


Jupyter 服务器环境
------------------

我们的操作系统通常设为 Linux (Mac 亦可)．通常这会不太方便，因为主要操作系统一般是 Windows，因此我们会期望将 Python 部署在远程 Linux 服务器上．而我们又会大量使用 Jupyter Notebook，其默认使用的地址是服务器的本地地址 (:code:`127.0.0.1:8080`)；而这对于本地电脑而言是不可访问的．因此，需要对 Jupyter Notebook 的地址进行更改，才能让本地电脑访问服务器所启动的 Jupyter Notebook．这里参考 Jupyter Notebook 的官方文档 `Running a notebook server <https://jupyter-notebook.readthedocs.io/en/stable/public_server.html>`_ 讲述如何配置 Jupyter Notebook．

   .. tip::
      如果 Python 可以部署在本地电脑，或者可以使用 WSL (Windows Subsystem of Linux)，这一节可以跳过．

#. 首先执行下述语句：
   
   .. code-block:: bash
   
      $ jupyter notebook --generate-config

   这将产生 Jupyter Notebook 的配置文件 :code:`$HOME/.jupyter/jupyter_notebook_config.py`

#. 在 Jupyter Notebook 配置文件中，你将看到下述语句：
   ::

      #c.NotebookApp.ip = 'localhost'

   对上述语句取消注释，并将其中的 :code:`localhost` 更改为服务器的 IP 地址．Jupyter Notebook 的服务器环境就设立好了．

#. 我们可以试一下 Jupyter Notebook 了．在 Bash 下执行
   
   .. code-block:: bash

      $ jupyter notebook --no-browser

   将会弹出一些输出．我们关心下述输出
   
   .. code-block:: text

      Copy/paste this URL into your browser when you connect for the first time,
      to login with a token:

   后面一行的地址；将该地址复制到本地计算机的浏览器中，就可以使用服务器的 Jupyter Notebook 了．

.. 
      Psi4 环境
      ---------

      在这份教程中，我们不需要改动 Psi4 的程序，也不需要进行编译，只需要获得其 Python 版本的二进制可执行文件即可．这里的安装过程主要参考 `Psi4NumPy <https://github.com/psi4/psi4numpy>`_ 上的说明．

      #. 在安装完 conda 或 Anaconda 后，执行

      .. code-block:: bash

            $ conda create -n p4env psi4 -c psi4/label/dev

      .. attention::
            一方面，我们需要使用 DFT 模块，因此需要下载 :code:`psi4/label/dev` 而并非 :code:`psi4` 的 Psi4 版本；

            另一方面，上述的命令是创建了一个虚拟环境，它是专门为 Psi4 创建的环境．这么做是因为避免与最新版本的 Anaconda 产生库的依赖冲突，保证默认的 Python 比较干净．因此，这里没有直接在默认的 Python 环境下安装 Psi4．这样做多少会对使用产生不便，但避免库依赖关系混乱可能导致的更严重的问题．

      #. 在每次需要使用 Psi4 或维护其库依赖关系时，需要在 Bash 下执行

      .. code-block:: bash

            $ source activate p4env

      当 Terminal 前有提示 :code:`(p4env)` 时，即意味着进入 Psi4 的虚拟环境了．以后我们假设所有的命令都在该虚拟环境下执行．

      #. Psi4 的 Python 二进制文件已经可以使用了；但 Jupyter 与 MatPlotLib 并不在其依赖关系中；而这些库是我们需要的．因此，我们需要在 Psi4 的虚拟环境下执行

      .. code-block:: bash

            (p4env) $ conda install jupyter matplotlib

      #. 至此我们已经完成了 Psi4 的安装．Psi4 可以作为一个量化软件，也可以作为 Python API 使用．对于前者，我们可以简单地使用一个输入文件作测试：:download:`input.dat <include/input.dat>`

      在 Bash 下使用下述命令进行测试：

      .. code-block:: bash

            (p4env) $ psi4 input.dat

      如果能正常地看到 :code:`output.dat` 且有正常的输出信息，即表明安装正常．

      #. 我们也可以尝试在 Python 下做一个小测试；如果看到与下述输出一样的信息，则表明 Python API 可以正常调用：
      ::

            >>> import psi4
            >>> mol = psi4.geometry("""
            ...     O  0.000    -0.000    -0.079
            ...     H  0.000     0.707     0.628
            ...     H  0.000    -0.707     0.628
            ...     symmetry c1
            ... """)
            >>> psi4.set_options({'basis': '6-31g'})
            >>> psi4.core.set_output_file('output.dat', False)
            >>> scf_e, scf_wfn = psi4.energy('B3LYP', return_wfn=True)
            >>> scf_e
            -76.3771897718305