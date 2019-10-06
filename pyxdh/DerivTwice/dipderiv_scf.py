import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipDerivSCF(DerivTwiceSCF):

    def _get_H_2_ao(self):
        mol = self.mol
        natm, nao = mol.natm, mol.nao
        mol_slice = self.A.mol_slice
        int1e_irp = mol.intor("int1e_irp").reshape(3, 3, nao, nao)
        H_2_ao = np.zeros((3, natm, 3, nao, nao))
        for A in range(natm):
            sA = mol_slice(A)
            H_2_ao[:, A, :, :, sA] = int1e_irp[:, :, :, sA]
        H_2_ao += H_2_ao.swapaxes(-1, -2)
        return H_2_ao.reshape((3, 3 * natm, nao, nao))

    def _get_S_2_ao(self):
        return 0

    def _get_F_2_ao_JKcontrib(self):
        return 0, 0

    def _get_F_2_ao_GGAcontrib(self):
        return 0

    def _get_eri2_ao(self):
        return 0

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return np.einsum("ABuv, uv -> AB", self.H_2_ao, self.A.D)

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        return 4 * np.einsum("Api, Bpi -> AB", A.H_1_mo[:, :, so], B.U_1[:, :, so])

    def _get_E_2(self):
        mol = self.mol
        natm = mol.natm
        dipderiv_nuc = np.zeros((3, natm, 3))
        for A in range(natm):
            dipderiv_nuc[:, A, :] = np.eye(3) * mol.atom_charge(A)
        dipderiv_nuc.shape = (3, 3 * natm)
        return self._get_E_2_Skeleton() + self._get_E_2_U() + dipderiv_nuc


class DipDerivNCDFT(DerivTwiceNCDFT, DipDerivSCF):

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        E_2_U = 4 * np.einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * np.einsum("ABai, ai -> AB", self.pdB_F_A_mo[:, :, sv, so], self.Z)
        return E_2_U


class Test_DipDerivSCF:

    @staticmethod
    def valid_assert(dip_helper, grad_helper, resource_path):
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface
        dipderiv_config = {"deriv_A": dip_helper, "deriv_B": grad_helper}
        helper = DipDerivSCF(dipderiv_config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", resource_path))
        assert (np.allclose(E_2.T, formchk.dipolederiv(), atol=1e-6, rtol=1e-4))

    def test_SCF_dipderiv(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.DerivOnce import DipoleSCF, GradSCF

        # HF
        H2O2 = Mol_H2O2()
        dip_deriv = DipoleSCF({"scf_eng": H2O2.hf_eng, "cphf_tol": 1e-10})
        grad_deriv = GradSCF({"scf_eng": H2O2.hf_eng, "cphf_tol": 1e-10})
        self.valid_assert(dip_deriv, grad_deriv, "Validation/gaussian/H2O2-HF-freq.fchk")

        # B3LYP
        H2O2 = Mol_H2O2()
        grids_cphf = H2O2.gen_grids(50, 194)
        dip_deriv = DipoleSCF({"scf_eng": H2O2.gga_eng, "cphf_grids": grids_cphf, "cphf_tol": 1e-10})
        grad_deriv = GradSCF({"scf_eng": H2O2.gga_eng, "cphf_grids": grids_cphf, "cphf_tol": 1e-10})
        self.valid_assert(dip_deriv, grad_deriv, "Validation/gaussian/H2O2-B3LYP-freq.fchk")

    def test_HF_B3LYP_dipderiv(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.DerivOnce import DipoleNCDFT, GradNCDFT
        import pickle

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng, "nc_eng": H2O2.gga_eng}
        dip_helper = DipoleNCDFT(config)
        grad_helper = GradNCDFT(config)
        config = {"deriv_A": dip_helper, "deriv_B": grad_helper}

        helper = DipDerivNCDFT(config)
        E_2 = helper.E_2

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/ncdft_dipderiv_hf_b3lyp.dat"), "rb") as f:
            ref_dipderiv = pickle.load(f)["dipderiv"]

        assert(np.allclose(E_2.T, ref_dipderiv, atol=1e-6, rtol=1e-4))
