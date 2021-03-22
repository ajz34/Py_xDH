# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceUSCF, DerivTwiceUMP2, HessSCF
from pyxdh.Utilities import timing, cached_property
# pytest
from pyscf import gto, scf
from pyxdh.DerivOnce import GradUSCF
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename


class HessUSCF(DerivTwiceUSCF, HessSCF):

    @cached_property
    @timing
    def F_2_ao_JKcontrib(self):
        D = self.D
        eri2_ao = self.eri2_ao
        return (
            einsum("ABuvkl, xkl -> ABuv", eri2_ao, D)[None, :].repeat(2, axis=0),
            einsum("ABukvl, xkl -> xABuv", eri2_ao, D),
        )

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        D = self.D
        cx = self.cx if cx is None else cx
        # HF Contribution
        E_SS_HF_contrib = (
            + einsum("ABuv, xuv -> AB", self.H_2_ao, D)
            + 0.5 * einsum("xABuv, xuv -> AB", self.F_2_ao_Jcontrib - cx * self.F_2_ao_Kcontrib, D)
        )
        return E_SS_HF_contrib


class HessUMP2(DerivTwiceUMP2, HessUSCF):
    pass


class TestHessU:
    
    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()
    
    def test_u_uhf_hess(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
        gradh = GradUSCF({"scf_eng": scf_eng})
        hessh = HessUSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)

    # TODO: UMP2 Hessian is possibly flawed
    # def test_u_mp2_hess(self):
    #     scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
    #     gradh = GradUMP2({"scf_eng": scf_eng, "cphf_tol": 1e-10})
    #     hessh = HessUMP2({"deriv_A": gradh})
    #     formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
    #     assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-5, rtol=1e-4)

