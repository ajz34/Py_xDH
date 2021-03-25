# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC, abstractmethod
from functools import partial
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceDFSCF
from pyxdh.DerivTwice import DerivTwiceSCF
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

