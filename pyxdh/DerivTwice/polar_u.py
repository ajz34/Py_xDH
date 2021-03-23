# basic utilities
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceUSCF, PolarSCF, DerivTwiceUMP2


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
