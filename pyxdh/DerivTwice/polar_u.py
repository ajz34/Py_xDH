# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceUSCF, PolarSCF, DerivTwiceUMP2
# pytest
from pyscf import gto, scf
from pyxdh.DerivOnce import DipoleUSCF, DipoleUMP2
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename


class PolarUSCF(DerivTwiceUSCF, PolarSCF):
    pass


class PolarUMP2(DerivTwiceUMP2, PolarUSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + einsum("xBpq, xApq -> AB", B.pdA_D_r_oovv, A.B_1)
            + einsum("Aai, Bai -> AB", A.U_1[0, :, sv[0], so[0]], self.RHS_B[0])
            + einsum("Aai, Bai -> AB", A.U_1[1, :, sv[1], so[1]], self.RHS_B[1])
            + einsum("xpq, xABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class TestPolarU:

    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()

    def test_u_uhf_polar(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
        diph = DipoleUSCF({"scf_eng": scf_eng})
        polh = PolarUSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_u_mp2_polar(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
        diph = DipoleUMP2({"scf_eng": scf_eng, "cphf_tol": 1e-12})
        polh = PolarUMP2({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)
