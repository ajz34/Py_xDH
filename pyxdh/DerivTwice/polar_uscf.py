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
        from pyxdh.DerivOnce import DipoleUSCF

        CH3 = Mol_CH3()
        config = {
            "scf_eng": CH3.hf_eng,
            "cphf_tol": 1e-10,
        }
        dip_helper = DipoleUSCF(config)
        config = {
            "deriv_A": dip_helper,
            "deriv_B": dip_helper,
            "cphf_tol": 1e-10,
        }
        helper = PolarUSCF(config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))

        assert(np.allclose(
            -E_2, formchk.polarizability(),
            atol=1e-5, rtol=1e-4
        ))


