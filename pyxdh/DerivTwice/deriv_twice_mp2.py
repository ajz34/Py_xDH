import numpy as np
from abc import ABC
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT
from pyxdh.DerivOnce import DerivOnceMP2, DerivOnceXDH

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: C1
class DerivTwiceMP2(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceMP2, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceMP2 classes
        self.A = config["deriv_A"]  # type: DerivOnceMP2
        self.B = config["deriv_B"]  # type: DerivOnceMP2
        assert(isinstance(self.A, DerivOnceMP2))
        assert(isinstance(self.B, DerivOnceMP2))
        assert(self.A.cc == self.B.cc)
        self.cc = self.A.cc

        # For simplicity, these values are not set to be properties
        # However, these values should not be changed or redefined
        self.t_iajb = self.A.t_iajb
        self.T_iajb = self.A.T_iajb
        self.L = self.A.L
        self.D_r = self.A.D_r
        self.W_I = self.A.W_I
        self.D_iajb = self.A.D_iajb

        # For MP2 calculation
        self._pdB_D_r_oovv = NotImplemented
        self._RHS_B = NotImplemented
        self._pdB_W_I = NotImplemented

    # region Properties

    @property
    def pdB_D_r_oovv(self):
        if self._pdB_D_r_oovv is NotImplemented:
            self._pdB_D_r_oovv = self._get_pdB_D_r_oovv()
        return self._pdB_D_r_oovv

    @property
    def pdB_W_I(self):
        if self._pdB_W_I is NotImplemented:
            self._pdB_W_I = self._get_pdB_W_I()
        return self._pdB_W_I

    # endregion

    # region Functions

    def _get_pdB_D_r_oovv(self):
        B = self.B
        so, sv = self.so, self.sv
        nmo = self.nmo

        pdB_D_r_oovv = np.zeros((B.pdA_t_iajb.shape[0], nmo, nmo))
        pdB_D_r_oovv[:, so, so] -= 2 * np.einsum("iakb, Ajakb -> Aij", B.T_iajb, B.pdA_t_iajb)
        pdB_D_r_oovv[:, sv, sv] += 2 * np.einsum("iajc, Aibjc -> Aab", B.T_iajb, B.pdA_t_iajb)
        pdB_D_r_oovv[:, so, so] -= 2 * np.einsum("Aiakb, jakb -> Aij", B.pdA_T_iajb, B.t_iajb)
        pdB_D_r_oovv[:, sv, sv] += 2 * np.einsum("Aiajc, ibjc -> Aab", B.pdA_T_iajb, B.t_iajb)

        return pdB_D_r_oovv

    def _get_RHS_B(self):
        B = self.B
        so, sv, sa = self.so, self.sv, self.sa
        U_1, D_r, pdB_F_0_mo, eri0_mo = B.U_1, B.D_r, B.pdA_F_0_mo, B.eri0_mo
        Ax0_Core, Ax1_Core = B.Ax0_Core, B.Ax1_Core
        pdB_D_r_oovv = self.pdB_D_r_oovv

        # total partial eri0
        pdA_eri0_mo = B.pdA_eri0_mo

        # D_r Part
        RHS_B = Ax0_Core(sv, so, sa, sa)(pdB_D_r_oovv)
        RHS_B += Ax1_Core(sv, so, sa, sa)(D_r)
        RHS_B += np.einsum("Apa, pi -> Aai", U_1[:, :, sv], Ax0_Core(sa, so, sa, sa)(D_r))
        RHS_B += np.einsum("Api, ap -> Aai", U_1[:, :, so], Ax0_Core(sv, sa, sa, sa)(D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(np.einsum("Amp, pq -> Amq", U_1, D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(np.einsum("Amq, pq -> Apm", U_1, D_r))
        # (ea - ei) * Dai
        RHS_B += np.einsum("Aca, ci -> Aai", pdB_F_0_mo[:, sv, sv], D_r[sv, so])
        RHS_B -= np.einsum("Aki, ak -> Aai", pdB_F_0_mo[:, so, so], D_r[sv, so])
        # 2-pdm part
        RHS_B -= 4 * np.einsum("Ajakb, ijbk -> Aai", B.pdA_T_iajb, eri0_mo[so, so, sv, so])
        RHS_B += 4 * np.einsum("Aibjc, abjc -> Aai", B.pdA_T_iajb, eri0_mo[sv, sv, so, sv])
        RHS_B -= 4 * np.einsum("jakb, Aijbk -> Aai", B.T_iajb, pdA_eri0_mo[:, so, so, sv, so])
        RHS_B += 4 * np.einsum("ibjc, Aabjc -> Aai", B.T_iajb, pdA_eri0_mo[:, sv, sv, so, sv])

        return RHS_B

    def _get_pdB_W_I(self):
        so, sv = self.so, self.sv
        natm, nmo = self.natm, self.nmo
        pdA_T_iajb, T_iajb = self.B.pdA_T_iajb, self.B.T_iajb
        eri0_mo, pdA_eri0_mo = self.B.eri0_mo, self.B.pdA_eri0_mo

        pdR_W_I = np.zeros((natm * 3, nmo, nmo))
        pdR_W_I[:, so, so] -= 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb, eri0_mo[so, sv, so, sv])
        pdR_W_I[:, sv, sv] -= 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb, eri0_mo[so, sv, so, sv])
        pdR_W_I[:, sv, so] -= 4 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb, eri0_mo[so, so, sv, so])
        pdR_W_I[:, so, so] -= 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb, pdA_eri0_mo[:, so, sv, so, sv])
        pdR_W_I[:, sv, sv] -= 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb, pdA_eri0_mo[:, so, sv, so, sv])
        pdR_W_I[:, sv, so] -= 4 * np.einsum("jakb, Aijbk -> Aai", T_iajb, pdA_eri0_mo[:, so, so, sv, so])

        return pdR_W_I

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        eri1_mo, U_1 = A.eri1_mo, B.U_1
        #
        pdB_pdpA_eri0_iajb = (
            + np.einsum("ABuvkl, up, vq, kr, ls -> ABpqrs", self.eri2_ao, self.Co, self.Cv, self.Co, self.Cv)
            + np.einsum("Apjkl, Bpi -> ABijkl", eri1_mo[:, :, sv, so, sv], U_1[:, :, so])
            + np.einsum("Aipkl, Bpj -> ABijkl", eri1_mo[:, so, :, so, sv], U_1[:, :, sv])
            + np.einsum("Aijpl, Bpk -> ABijkl", eri1_mo[:, so, sv, :, sv], U_1[:, :, so])
            + np.einsum("Aijkp, Bpl -> ABijkl", eri1_mo[:, so, sv, so, :], U_1[:, :, sv])
        )

        E_2_MP2_Contrib = (
            # D_r * B
            + np.einsum("pq, ABpq -> AB", self.D_r, self.pdB_B_A)
            + np.einsum("Bpq, Apq -> AB", self.pdB_D_r_oovv, A.B_1)
            + np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            # W_I * S
            + np.einsum("pq, ABpq -> AB", self.W_I, self.pdB_S_A_mo)
            + np.einsum("Bpq, Apq -> AB", self.pdB_W_I, A.S_1_mo)
            # T * g
            + 2 * np.einsum("Biajb, Aiajb -> AB", B.pdA_T_iajb, A.eri1_mo[:, so, sv, so, sv])
            + 2 * np.einsum("iajb, ABiajb -> AB", self.T_iajb, pdB_pdpA_eri0_iajb)
        )
        return E_2_MP2_Contrib

    def _get_E_2(self):
        return super(DerivTwiceMP2, self)._get_E_2() + self._get_E_2_MP2_Contrib()

    # endregion


class DerivTwiceXDH(DerivTwiceMP2, DerivTwiceNCDFT, ABC):

    def __init__(self, config):
        super(DerivTwiceXDH, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceXDH classes
        self.A = config["deriv_A"]  # type: DerivOnceXDH
        self.B = config["deriv_B"]  # type: DerivOnceXDH
        assert(isinstance(self.A, DerivOnceXDH))
        assert(isinstance(self.B, DerivOnceXDH))

    def _get_RHS_B(self):
        RHS_B = DerivTwiceMP2._get_RHS_B(self)
        RHS_B += 4 * self.B.pdA_nc_F_0_mo[:, self.sv, self.so]
        return RHS_B

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        E_2_U = 4 * np.einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U -= 2 * np.einsum("Aki, Bki -> AB", A.S_1_mo[:, so, so], B.pdA_nc_F_0_mo[:, so, so])
        E_2_U -= 2 * np.einsum("ABki, ki -> AB", self.pdB_S_A_mo[:, :, so, so], A.nc_deriv.F_0_mo[so, so])
        return E_2_U
