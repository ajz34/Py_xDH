import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceUSCF, PolarSCF
from pyxdh.Utilities import timing, GridIterator, KernelHelper


class PolarUSCF(DerivTwiceUSCF, PolarSCF):
    pass


class Test_PolarUSCF:

    def test_UHF_polar(self):
        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import GradUSCF

