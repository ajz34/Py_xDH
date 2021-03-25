# basic utilities
import numpy as np
from opt_einsum import contract as einsum
from scipy.linalg import solve_triangular
# python utilities
from abc import ABC, abstractmethod
from functools import partial
from warnings import warn
# pyscf utilities
from pyscf import gto
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceSCF, DerivOnceMP2
from pyxdh.Utilities import cached_property
# simplification
st = partial(solve_triangular, lower=True)

# additional
einsum = partial(einsum, optimize="greedy")



class DerivOnceDFSCF(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFSCF, self).__init__(config)
        self.scf_eng = self.scf_eng  # cheat pycharm to disable it's warning on type
        self.aux_jk = self.scf_eng.with_df.auxmol  # type: gto.Mole

    # region Auxiliary Basis Integral Generation

    @staticmethod
    def _gen_int2c2e(aux):
        return aux.intor("int2c2e")

    @staticmethod
    def _gen_int3c2e(mol, aux):
        return int3c_wrapper(mol, aux, "int3c2e", "s1")()

    @staticmethod
    def _gen_L_aux(int2c2e):  # in actual programming, L_aux may refers to L_ri or L_jk
        return np.linalg.cholesky(int2c2e)  # should be lower triangular

    @staticmethod
    def _gen_L_inv(L_aux):
        return np.linalg.inv(L_aux)

    @staticmethod
    def _gen_Y_ao(int3c2e, L_inv):
        return einsum("μνQ, PQ -> μνP", int3c2e, L_inv)

    @staticmethod
    @abstractmethod
    def _gen_int2c2e_1(aux):
        pass

    @staticmethod
    @abstractmethod
    def _gen_int3c2e_1(mol, aux):
        pass

    @staticmethod
    def _gen_L_1(L, L_inv, int2c2e_1):
        if not isinstance(int2c2e_1, np.ndarray):
            return 0
        l = np.zeros_like(L)
        for i in range(l.shape[0]):
            l[i, :i] = 1
            l[i, i] = 1/2
        # return L @ (l * (L_inv @ int2c2e_1 @ L_inv.T))   # greedy search is required
        return einsum("PR, RQ, RS, AST, QT -> APQ", L, l, L_inv, int2c2e_1, L_inv, optimize="greedy")

    @staticmethod
    def _gen_L_inv_1(L_inv, L_1):
        if not isinstance(L_1, np.ndarray):
            return 0
        return - einsum("PR, ARS, SQ -> APQ", L_inv, L_1, L_inv)

    @staticmethod
    def _gen_Y_ao_1(int3c2e, int3c2e_1, L_inv, L_inv_1):
        Y_ao_1 = 0
        if isinstance(int3c2e_1, np.ndarray):
            Y_ao_1 += einsum("AμνQ, PQ -> AμνP", int3c2e_1, L_inv)
        if isinstance(L_inv_1, np.ndarray):
            Y_ao_1 += einsum("μνQ, APQ -> AμνP", int3c2e, L_inv_1)
        return Y_ao_1

    # endregion Auxiliary Basis Integral Generation

    # region Auxiliary Basis JK

    @cached_property
    def int2c2e_jk(self):
        return self._gen_int2c2e(self.aux_jk)

    @cached_property
    def int3c2e_jk(self):
        return self._gen_int3c2e(self.mol, self.aux_jk)

    @cached_property
    def L_jk(self):
        return self._gen_L_aux(self.int2c2e_jk)

    @cached_property
    def L_inv_jk(self):
        return self._gen_L_inv(self.L_jk)

    @cached_property
    def Y_ao_jk(self):
        return self._gen_Y_ao(self.int3c2e_jk, self.L_inv_jk)

    @cached_property
    def int2c2e_1_jk(self):
        return self._gen_int2c2e_1(self.aux_jk)

    @cached_property
    def int3c2e_1_jk(self):
        return self._gen_int3c2e_1(self.mol, self.aux_jk)

    @cached_property
    def L_1_jk(self):
        return self._gen_L_1(self.L_jk, self.L_inv_jk, self.int2c2e_1_jk)

    @cached_property
    def L_inv_1_jk(self):
        return self._gen_L_inv_1(self.L_inv_jk, self.L_1_jk)

    @cached_property
    def Y_ao_1_jk(self):
        return self._gen_Y_ao_1(self.int3c2e_jk, self.int3c2e_1_jk, self.L_inv_jk, self.L_inv_1_jk)

    # endregion Auxiliary Basis JK

    @cached_property
    def eri0_ao(self):
        warn("eri0 should not be called in density fitting module!", FutureWarning)
        return einsum("μνP, PQ, κλQ -> μνκλ", self.int3c2e_jk, np.linalg.inv(self.int2c2e_jk), self.int3c2e_jk)


class DerivOnceDFMP2(DerivOnceMP2, DerivOnceDFSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFMP2, self).__init__(config)
        self.aux_ri = config["aux_ri"]  # type: gto.Mole

    @cached_property
    def int2c2e_ri(self):
        return self._gen_int2c2e(self.aux_ri)

    @cached_property
    def int3c2e_ri(self):
        return self._gen_int3c2e(self.mol, self.aux_ri)

    @cached_property
    def L_ri(self):
        return self._gen_L_aux(self.int2c2e_ri)

    @cached_property
    def L_inv_ri(self):
        return self._gen_L_inv(self.L_ri)

    @cached_property
    def Y_ao_ri(self):
        return self._gen_Y_ao(self.int3c2e_ri, self.L_inv_ri)

    @cached_property
    def Y_mo_ri(self):
        return einsum("μνP, μp, νq -> pqP", self.Y_ao_ri, self.C, self.C)

    @property
    def Y_ia_ri(self):
        return self.Y_mo_ri[self.so, self.sv, :]

    @cached_property
    def int2c2e_1_ri(self):
        return self._gen_int2c2e_1(self.aux_ri)

    @cached_property
    def int3c2e_1_ri(self):
        return self._gen_int3c2e_1(self.mol, self.aux_ri)

    @cached_property
    def L_1_ri(self):
        return self._gen_L_1(self.L_ri, self.L_inv_ri, self.int2c2e_1_ri)

    @cached_property
    def L_inv_1_ri(self):
        return self._gen_L_inv_1(self.L_inv_ri, self.L_1_ri)

    @cached_property
    def Y_ao_1_ri(self):
        return self._gen_Y_ao_1(self.int3c2e_ri, self.int3c2e_1_ri, self.L_inv_ri, self.L_inv_1_ri)

    @cached_property
    def Y_mo_1_ri(self):
        return einsum("AμνP, μp, νq -> ApqP", self.Y_ao_1_ri, self.C, self.C)

    @property
    def Y_ia_1_ri(self):
        return self.Y_mo_1_ri[:, self.so, self.sv, :]

    @cached_property
    def t_iajb(self):
        return einsum("iaP, jbP -> iajb", self.Y_ia_ri, self.Y_ia_ri) / self.D_iajb

    def _get_L(self):
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        T_iajb = self.T_iajb
        Y_ia_ri, Y_mo_ri = self.Y_ia_ri, self.Y_mo_ri
        L = np.zeros((nvir, nocc))
        L += Ax0_Core(sv, so, sa, sa)(self.D_r_oovv)
        L -= 4 * np.einsum("jakb, ijP, kbP -> ai", T_iajb, Y_mo_ri[so, so], Y_ia_ri)
        L += 4 * np.einsum("ibjc, abP, jcP -> ai", T_iajb, Y_mo_ri[sv, sv], Y_ia_ri)
        return L

    @cached_property
    def W_I(self):
        so, sv = self.so, self.sv
        nmo = self.nmo
        T_iajb = self.T_iajb
        Y_ia_ri, Y_mo_ri = self.Y_ia_ri, self.Y_mo_ri
        W_I = np.zeros((nmo, nmo))
        W_I[so, so] = - 2 * einsum("iakb, jaP, kbP -> ij", T_iajb, Y_ia_ri, Y_ia_ri)
        W_I[sv, sv] = - 2 * einsum("iajc, ibP, jcP -> ab", T_iajb, Y_ia_ri, Y_ia_ri)
        W_I[sv, so] = - 4 * einsum("jakb, ijP, kbP -> ai", T_iajb, Y_mo_ri[so, so], Y_ia_ri)
        return W_I
