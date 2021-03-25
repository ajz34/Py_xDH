# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC, abstractmethod
from functools import partial
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceDFSCF, DerivOnceDFMP2
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceMP2
from pyxdh.Utilities import cached_property

# additional
einsum = partial(einsum, optimize="greedy")


class DerivTwiceDFSCF(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceDFSCF, self).__init__(config)
        self.A = self.A  # type: DerivOnceDFSCF
        self.B = self.B  # type: DerivOnceDFSCF
        self.aux_jk = self.A.aux_jk

    # region Auxiliary Basis Integral Generation

    @staticmethod
    @abstractmethod
    def _gen_int2c2e_2(aux):
        pass

    @staticmethod
    @abstractmethod
    def _gen_int3c2e_2(mol, aux):
        pass

    @staticmethod
    def _gen_L_2(L, L_inv, int2c2e_A, int2c2e_B, int2c2e_2):
        if not isinstance(int2c2e_A, np.ndarray) and isinstance(int2c2e_B, np.ndarray):
            return 0
        l = np.zeros_like(L)
        for i in range(l.shape[0]):
            l[i, :i] = 1
            l[i, i] = 1 / 2
        T_A = L_inv @ int2c2e_A @ L_inv.T
        T_B = L_inv @ int2c2e_B @ L_inv.T
        L_2 = (
            + einsum("PR, BRS, ASQ -> ABPQ", L, l * T_B, l * T_A)
            - einsum("PR, RQ, BRS, ASQ -> ABPQ", L, l, l * T_B, T_A)
            + einsum("PR, RQ, RS, ABST, QT -> ABPQ", L, l, L_inv, int2c2e_2, L_inv)
            - einsum("PR, RQ, ARS, BQS -> ABPQ", L, l, T_A, l * T_B)
        )
        return L_2

    @staticmethod
    def _gen_L_inv_2(L, L_inv, int2c2e_A, int2c2e_B, int2c2e_2):
        if not isinstance(int2c2e_A, np.ndarray) and isinstance(int2c2e_B, np.ndarray):
            return 0
        l = np.zeros_like(L)
        for i in range(l.shape[0]):
            l[i, :i] = 1
            l[i, i] = 1 / 2
        T_A = L_inv @ int2c2e_A @ L_inv.T
        T_B = L_inv @ int2c2e_B @ L_inv.T
        L_inv_2 = (
            - einsum("PR, PS, ABST, RT, RQ -> ABPQ", l, L_inv, int2c2e_2, L_inv, L_inv)
            + einsum("PR, BPS, ASR, RQ -> ABPQ", l, l * T_B, T_A, L_inv)
            + einsum("PR, APS, BRS, RQ -> ABPQ", l, T_A, l * T_B, L_inv)
            + einsum("APR, BRS, SQ -> ABPQ", l * T_A, l * T_B, L_inv)
        )
        return L_inv_2

    @staticmethod
    def _gen_Y_ao_2(L_inv, L_inv_A, L_inv_B, L_inv_2, int3c2e, int3c2e_A, int3c2e_B, int3c2e_2):
        Y_ao_2 = 0
        if isinstance(int3c2e_2, np.ndarray):
            Y_ao_2 += einsum("ABμνQ, PQ -> ABμνP", int3c2e_2, L_inv)
        if isinstance(int3c2e_A, np.ndarray) and isinstance(L_inv_B, np.ndarray):
            Y_ao_2 += einsum("AμνQ, BPQ -> ABμνP", int3c2e_A, L_inv_B)
        if isinstance(int3c2e_B, np.ndarray) and isinstance(L_inv_A, np.ndarray):
            Y_ao_2 += einsum("BμνQ, APQ -> ABμνP", int3c2e_B, L_inv_A)
        if isinstance(L_inv_2, np.ndarray):
            Y_ao_2 += einsum("μνQ, ABPQ -> ABμνP", int3c2e, L_inv_2)
        return Y_ao_2

    # endregion Auxiliary Basis Integral Generation

    # region Auxiliary Basis JK

    @cached_property
    def int2c2e_2_jk(self):
        return self._gen_int2c2e_2(self.aux_jk)

    @cached_property
    def int3c2e_2_jk(self):
        return self._gen_int3c2e_2(self.mol, self.aux_jk)

    @cached_property
    def L_2_jk(self):
        return self._gen_L_2(self.A.L_jk, self.A.L_inv_jk, self.A.int2c2e_1_jk, self.B.int2c2e_1_jk, self.int2c2e_2_jk)

    @cached_property
    def L_inv_2_jk(self):
        return self._gen_L_inv_2(self.A.L_jk, self.A.L_inv_jk, self.A.int2c2e_1_jk, self.B.int2c2e_1_jk, self.int2c2e_2_jk)

    @cached_property
    def Y_ao_2_jk(self):
        return self._gen_Y_ao_2(
            self.A.L_inv_jk, self.A.L_inv_1_jk, self.B.L_inv_1_jk, self.L_inv_2_jk,
            self.A.int3c2e_jk, self.A.int3c2e_1_jk, self.B.int3c2e_1_jk, self.int3c2e_2_jk)

    # endregion Auxiliary Basis JK


class DerivTwiceDFMP2(DerivTwiceMP2, DerivTwiceDFSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceDFMP2, self).__init__(config)
        # Only cheat IDE
        self.A = self.A  # type: DerivOnceDFMP2
        self.B = self.B  # type: DerivOnceDFMP2
        self.aux_jk = self.A.aux_jk
        self.aux_ri = self.A.aux_ri

    # region Auxiliary Basis RI

    @cached_property
    def int2c2e_2_ri(self):
        return self._gen_int2c2e_2(self.aux_ri)

    @cached_property
    def int3c2e_2_ri(self):
        return self._gen_int3c2e_2(self.mol, self.aux_ri)

    @cached_property
    def L_2_ri(self):
        return self._gen_L_2(self.A.L_ri, self.A.L_inv_ri, self.A.int2c2e_1_ri, self.B.int2c2e_1_ri, self.int2c2e_2_ri)

    @cached_property
    def L_inv_2_ri(self):
        return self._gen_L_inv_2(self.A.L_ri, self.A.L_inv_ri, self.A.int2c2e_1_ri, self.B.int2c2e_1_ri, self.int2c2e_2_ri)

    @cached_property
    def Y_ao_2_ri(self):
        return self._gen_Y_ao_2(
            self.A.L_inv_ri, self.A.L_inv_1_ri, self.B.L_inv_1_ri, self.L_inv_2_ri,
            self.A.int3c2e_ri, self.A.int3c2e_1_ri, self.B.int3c2e_1_ri, self.int3c2e_2_ri)

    @cached_property
    def Y_mo_2_ri(self):
        return einsum("ABμνP, μp, νq -> ABpqP", self.Y_ao_2_ri, self.C, self.C)

    @cached_property
    def Y_ia_2_ri(self):
        return self.Y_mo_2_ri[:, :, self.so, self.sv, :]

    # endregion Auxiliary Basis RI

    @cached_property
    def pdB_pdpA_eri0_iajb(self):
        raise RuntimeError("eri0 derivative should not be called in DF routines!")

    def _get_RHS_B(self):
        B = self.B
        so, sv, sa = self.so, self.sv, self.sa
        U_1, D_r, pdB_F_0_mo = B.U_1, B.D_r, B.pdA_F_0_mo
        Ax0_Core, Ax1_Core = B.Ax0_Core, B.Ax1_Core
        pdB_D_r_oovv = B.pdA_D_r_oovv

        Y_mo_ri, Y_ia_ri = B.Y_mo_ri, B.Y_ia_ri
        pdA_Y_mo_1_ri = B.pdA_Y_mo_1_ri
        pdA_Y_ia_1_ri = pdA_Y_mo_1_ri[:, so, sv, :]

        RHS_B = np.zeros((U_1.shape[0], self.nvir, self.nocc))
        # D_r Part
        RHS_B += Ax0_Core(sv, so, sa, sa)(pdB_D_r_oovv)
        RHS_B += Ax1_Core(sv, so, sa, sa)(D_r)
        RHS_B += einsum("Apa, pi -> Aai", U_1[:, :, sv], Ax0_Core(sa, so, sa, sa)(D_r))
        RHS_B += einsum("Api, ap -> Aai", U_1[:, :, so], Ax0_Core(sv, sa, sa, sa)(D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(einsum("Amp, pq -> Amq", U_1, D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(einsum("Amq, pq -> Apm", U_1, D_r))
        # (ea - ei) * Dai
        RHS_B += einsum("Aca, ci -> Aai", pdB_F_0_mo[:, sv, sv], D_r[sv, so])
        RHS_B -= einsum("Aki, ak -> Aai", pdB_F_0_mo[:, so, so], D_r[sv, so])
        # 2-pdm part
        RHS_B -= 4 * einsum("Ajakb, ijP, kbP -> Aai", B.pdA_T_iajb, Y_mo_ri[so, so], Y_ia_ri)
        RHS_B += 4 * einsum("Aibjc, abP, jcP -> Aai", B.pdA_T_iajb, Y_mo_ri[sv, sv], Y_ia_ri)
        RHS_B -= 4 * einsum("jakb, AijP, kbP -> Aai", B.T_iajb, pdA_Y_mo_1_ri[:, so, so], Y_ia_ri)
        RHS_B -= 4 * einsum("jakb, ijP, AkbP -> Aai", B.T_iajb, Y_mo_ri[so, so], pdA_Y_ia_1_ri)
        RHS_B += 4 * einsum("ibjc, AabP, jcP -> Aai", B.T_iajb, pdA_Y_mo_1_ri[:, sv, sv], Y_ia_ri)
        RHS_B += 4 * einsum("ibjc, abP, AjcP -> Aai", B.T_iajb, Y_mo_ri[sv, sv], pdA_Y_ia_1_ri)

        return RHS_B

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        pdB_pdpA_Y_ia_ri = (
            + self.Y_mo_2_ri[:, :, so, sv]
            + einsum("Bmi, AmaP -> ABiaP", B.U_1[:, :, so], A.Y_mo_1_ri[:, :, sv])
            + einsum("Bma, AimP -> ABiaP", B.U_1[:, :, sv], A.Y_mo_1_ri[:, so])
        )

        E_2_MP2_Contrib = (
            # D_r * B
            + einsum("pq, ABpq -> AB", self.D_r, self.pdB_B_A)
            + einsum("Bpq, Apq -> AB", B.pdA_D_r_oovv, A.B_1)
            + einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            # W_I * S
            + einsum("pq, ABpq -> AB", self.W_I, self.pdB_S_A_mo)
            + einsum("Bpq, Apq -> AB", B.pdA_W_I, A.S_1_mo)
            # T * g
            + 2 * einsum("Biajb, AiaP, jbP -> AB", B.pdA_T_iajb, A.Y_ia_1_ri, A.Y_ia_ri)
            + 2 * einsum("Biajb, iaP, AjbP -> AB", B.pdA_T_iajb, A.Y_ia_ri, A.Y_ia_1_ri)
            + 2 * einsum("iajb, AiaP, BjbP -> AB", self.T_iajb, A.Y_ia_1_ri, B.pdA_Y_mo_1_ri[:, so, sv])
            + 2 * einsum("iajb, ABiaP, jbP -> AB", self.T_iajb, pdB_pdpA_Y_ia_ri, A.Y_ia_ri)
            + 2 * einsum("iajb, BiaP, AjbP -> AB", self.T_iajb, B.pdA_Y_mo_1_ri[:, so, sv], A.Y_ia_1_ri)
            + 2 * einsum("iajb, iaP, ABjbP -> AB", self.T_iajb, A.Y_ia_ri, pdB_pdpA_Y_ia_ri)
        )
        return E_2_MP2_Contrib

    def _get_E_2(self):
        return super(DerivTwiceDFMP2, self)._get_E_2() + self._get_E_2_MP2_Contrib()
