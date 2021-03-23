# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT, DerivTwiceMP2, DerivTwiceXDH
from pyxdh.Utilities import cached_property


class DipDerivSCF(DerivTwiceSCF):

    @property
    def A_is_B(self) -> bool:
        return False

    @cached_property
    def H_2_ao(self):
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
    def eri2_ao(self):
        return 0

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return einsum("ABuv, uv -> AB", self.H_2_ao, self.A.D)

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        return 4 * einsum("Api, Bpi -> AB", A.H_1_mo[:, :, so], B.U_1[:, :, so])

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
        E_2_U = 4 * einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * einsum("ABai, ai -> AB", self.pdB_F_A_mo[:, :, sv, so], self.Z)
        return E_2_U


class DipDerivMP2(DerivTwiceMP2, DipDerivSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + np.einsum("Bpq, Apq -> AB", B.pdA_D_r_oovv, A.B_1)
            + np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            + np.einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class DipDerivXDH(DerivTwiceXDH, DipDerivMP2, DipDerivNCDFT):

    def _get_E_2_U(self):
        return DipDerivMP2._get_E_2_U(self)


