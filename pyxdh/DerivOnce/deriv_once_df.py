import numpy as np
import opt_einsum
from abc import ABC
from functools import partial
import os
from scipy.linalg import solve_triangular

from pyscf import gto
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper

from pyxdh.DerivOnce import DerivOnceSCF, DerivOnceMP2

MAXMEM = float(os.getenv("MAXMEM", 2))
st = partial(solve_triangular, lower=True)
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
einsum = opt_einsum.contract
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceDFSCF(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFSCF, self).__init__(config)
        self.aux_jk = self.scf_eng.with_df.auxmol  # type: gto.Mole

    def _get_eri0_ao(self):
        raise AssertionError("eri0 should not be called in density fitting module!")

    def _get_eri1_ao(self):
        raise AssertionError("eri1 should not be called in density fitting module!")

    @staticmethod
    def _get_int2c2e(aux):
        return aux.intor("int2c2e")

    @staticmethod
    def _get_int3c2e(mol, aux):
        return int3c_wrapper(mol, aux, "int3c2e", "s1")()


class DerivOnceDFMP2(DerivOnceMP2, DerivOnceDFSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFMP2, self).__init__(config)
        self.aux_ri = config["aux_ri"]  # type: gto.Mole
        self._int2c2e_ri = NotImplemented  # type: np.ndarray
        self._int3c2e_ri = NotImplemented  # type: np.ndarray
        self._L_ri = NotImplemented  # type: np.ndarray
        self._L_inv_ri = NotImplemented  # type: np.ndarray
        self._Y_ao_ri = NotImplemented  # type: np.ndarray
        self._Y_ia_ri = NotImplemented  # type: np.ndarray

    @property
    def int2c2e_ri(self):
        if self._int2c2e_ri is NotImplemented:
            return self.aux_ri.intor("int2c2e")
        return self._int2c2e_ri

    @property
    def int3c2e_ri(self):
        if self._int3c2e_ri is NotImplemented:
            return int3c_wrapper(self.mol, self.aux_ri, "int3c2e", "s1")()
        return self._int3c2e_ri

    @property
    def L_ri(self):
        if self._L_ri is NotImplemented:
            return np.linalg.cholesky(self.int2c2e_ri)
        return self._L_ri

    @property
    def L_inv_ri(self):
        if self._L_inv_ri is NotImplemented:
            return np.linalg.inv(self.L_ri)
        return self._L_inv_ri

    @property
    def Y_ao_ri(self):
        if self._Y_ao_ri is NotImplemented:
            return einsum("μνQ, PQ -> μνP", self.int3c2e_ri, self.L_inv_ri)
        return self._Y_ao_ri

    @property
    def Y_ia_ri(self):
        if self._Y_ia_ri is NotImplemented:
            return einsum("μνP, μi, νa -> iaP", self.Y_ao_ri, self.Co, self.Cv)
        return self._Y_ia_ri

    def _get_t_iajb(self):
        return einsum("iaP, jbP -> iajb", self.Y_ia_ri, self.Y_ia_ri) / self.D_iajb
