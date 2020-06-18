import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceUSCF, HessSCF
from pyxdh.Utilities import timing, GridIterator, KernelHelper

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class HessUSCF(DerivTwiceUSCF, HessSCF):

    @timing
    def _get_F_2_ao_JKcontrib(self):
        D = self.D
        eri2_ao = self.eri2_ao
        return (
            np.einsum("ABuvkl, xkl -> ABuv", eri2_ao, D)[None, :].repeat(2, axis=0),
            np.einsum("ABukvl, xkl -> xABuv", eri2_ao, D),
        )

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        D = self.D
        cx = self.cx if cx is None else cx
        # HF Contribution
        E_SS_HF_contrib = (
            + np.einsum("ABuv, xuv -> AB", self.H_2_ao, D)
            + 0.5 * np.einsum("xABuv, xuv -> AB", self.F_2_ao_Jcontrib - cx * self.F_2_ao_Kcontrib, D)
        )
        return E_SS_HF_contrib
    pass


class Test_HessUSCF:

    def test_UHF_hess(self):
        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import GradUSCF

        CH3 = Mol_CH3()
        config = {
            "scf_eng": CH3.hf_eng,
            "cphf_tol": 1e-10,
        }
        grad_helper = GradUSCF(config)
        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
            "cphf_tol": 1e-10,
        }
        helper = HessUSCF(config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-5, rtol=1e-4
        ))


