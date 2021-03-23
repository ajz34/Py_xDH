# basic utilities
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT, DerivTwiceMP2, DerivTwiceXDH
from pyxdh.Utilities import cached_property


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
            + einsum("Bpq, Apq -> AB", B.pdA_D_r_oovv, A.B_1)
            + einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            + einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class PolarXDH(DerivTwiceXDH, PolarMP2, PolarNCDFT):

    def _get_E_2_U(self):
        return PolarMP2._get_E_2_U(self)
