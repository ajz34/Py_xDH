# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT, DerivTwiceMP2, DerivTwiceXDH
from pyxdh.Utilities import cached_property
# pytest
from pyscf import gto, scf, dft
from pyxdh.DerivOnce import DipoleSCF, DipoleNCDFT, DipoleMP2, DipoleXDH
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface
import pickle


class PolarSCF(DerivTwiceSCF):

    @property
    def A_is_B(self) -> bool:
        return True

    @cached_property
    def H_2_ao(self):
        return 0

    @cached_property
    def S_2_ao(self):
        return 0

    @cached_property
    def F_2_ao_JKcontrib(self):
        return 0, 0

    @cached_property
    def F_2_ao_GGAcontrib(self):
        return 0

    @cached_property
    def F_2_mo(self):
        return 0

    @cached_property
    def eri2_ao(self):
        return 0

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return 0

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        return 4 * einsum("Api, Bpi -> AB", A.H_1_mo[:, :, so], B.U_1[:, :, so])

    def _get_E_2(self):
        return self._get_E_2_U()


class PolarNCDFT(DerivTwiceNCDFT, PolarSCF):

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        E_2_U = 4 * einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * einsum("ABai, ai -> AB", self.pdB_F_A_mo[:, :, sv, so], self.Z)
        return E_2_U


class PolarMP2(DerivTwiceMP2, PolarSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + einsum("Bpq, Apq -> AB", self.pdB_D_r_oovv, A.B_1)
            + einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            + einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class PolarXDH(DerivTwiceXDH, PolarMP2, PolarNCDFT):

    def _get_E_2_U(self):
        return PolarMP2._get_E_2_U(self)


class TestPolarR:

    mol = gto.Mole(atom="N 0. 0. 0.; H 1.5 0. 0.2; H 0.1 1.2 0.; H 0. 0. 1.", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.build()
    grids_cphf = dft.Grids(mol)
    grids_cphf.atom_grid = (50, 194)
    grids_cphf.build()

    def test_r_rhf_polar(self):
        scf_eng = scf.RHF(self.mol).run()
        diph = DipoleSCF({"scf_eng": scf_eng})
        polh = PolarSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-HF-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_b3lyp_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        diph = DipoleSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        polh = PolarSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-B3LYP-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_hfb3lyp_polar(self):
        scf_eng = scf.RHF(self.mol).run()
        nc_eng = dft.RKS(self.mol, xc="B3LYPg")
        nc_eng.grids = self.grids
        diph = DipoleNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng})
        polh = PolarNCDFT({"deriv_A": diph})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/NH3-HFB3LYP-pol.dat"), "rb") as f:
            ref_polar = pickle.load(f)
        # ASSERT: polar - numerical
        assert np.allclose(- polh.E_2, ref_polar, atol=1e-6, rtol=1e-4)

    def test_r_mp2_polar(self):
        scf_eng = scf.RHF(self.mol).run()
        diph = DipoleMP2({"scf_eng": scf_eng})
        polh = PolarMP2({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-MP2-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_xyg3_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf}
        diph = DipoleXDH(config)
        polh = PolarXDH({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYG3-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_xygjos_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        diph = DipoleXDH(config)
        polh = PolarXDH({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYGJOS-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)
