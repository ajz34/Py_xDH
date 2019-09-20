import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class PolarSCF(DerivTwiceSCF):

    def _get_H_2_ao(self):
        return 0

    def _get_S_2_ao(self):
        return 0

    def _get_F_2_ao_JKcontrib(self):
        return 0, 0

    def _get_F_2_ao_GGAcontrib(self):
        return 0

    def _get_F_2_mo(self):
        return 0

    def _get_eri2_ao(self):
        return 0

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return 0

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        return 4 * np.einsum("Api, Bpi -> AB", A.H_1_mo[:, :, so], B.U_1[:, :, so])

    def _get_E_2(self):
        return self._get_E_2_U()


class PolarNCDFT(DerivTwiceNCDFT, PolarSCF):

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        E_2_U = 4 * np.einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * np.einsum("ABai, ai -> AB", self.pdB_F_A_mo[:, :, sv, so], self.Z)
        return E_2_U


class Test_PolarSCF:

    @staticmethod
    def valid_assert(dip_config, resource_path):
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleSCF
        dip_helper = DipoleSCF(dip_config)
        polar_config = {"deriv_A": dip_helper, "deriv_B": dip_helper}
        helper = PolarSCF(polar_config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", resource_path))
        assert (np.allclose(- E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4))

    def test_SCF_polar(self):

        from pyxdh.Utilities.test_molecules import Mol_H2O2

        H2O2 = Mol_H2O2()
        grids_cphf = H2O2.gen_grids(50, 194)
        self.valid_assert({"scf_eng": H2O2.hf_eng}, "Validation/gaussian/H2O2-HF-freq.fchk")
        self.valid_assert({"scf_eng": H2O2.gga_eng, "cphf_grids": grids_cphf}, "Validation/gaussian/H2O2-B3LYP-freq.fchk")

    def test_HF_B3LYP_polar(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.DerivOnce import DipoleNCDFT
        import pickle

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng, "nc_eng": H2O2.gga_eng}
        dip_helper = DipoleNCDFT(config)
        config = {"deriv_A": dip_helper, "deriv_B": dip_helper}

        helper = PolarNCDFT(config)
        E_2 = helper.E_2

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/ncdft_polarizability_hf_b3lyp.dat"), "rb") as f:
            ref_polar = pickle.load(f)["polarizability"]

        assert(np.allclose(- E_2, ref_polar, atol=1e-6, rtol=1e-4))
