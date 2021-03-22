# basic utilities
import numpy as np
from opt_einsum import contract as einsum
from scipy.linalg import solve_triangular
# python utilities
from abc import ABC
from functools import partial
# pyscf utilities
from pyscf import gto
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceSCF, DerivOnceMP2
from pyxdh.Utilities import cached_property
# simplification
st = partial(solve_triangular, lower=True)


class DerivOnceDFSCF(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFSCF, self).__init__(config)
        self.aux_jk = self.scf_eng.with_df.auxmol  # type: gto.Mole

    @cached_property
    def eri0_ao(self):
        raise AssertionError("eri0 should not be called in density fitting module!")

    @cached_property
    def eri1_ao(self):
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

    @cached_property
    def int2c2e_ri(self):
        return self.aux_ri.intor("int2c2e")

    @cached_property
    def int3c2e_ri(self):
        return int3c_wrapper(self.mol, self.aux_ri, "int3c2e", "s1")()

    @cached_property
    def L_ri(self):
        return np.linalg.cholesky(self.int2c2e_ri)

    @cached_property
    def L_inv_ri(self):
        return np.linalg.inv(self.L_ri)

    @cached_property
    def Y_ao_ri(self):
        return einsum("μνQ, PQ -> μνP", self.int3c2e_ri, self.L_inv_ri)

    @cached_property
    def Y_ia_ri(self):
        return einsum("μνP, μi, νa -> iaP", self.Y_ao_ri, self.Co, self.Cv)

    @cached_property
    def t_iajb(self):
        return einsum("iaP, jbP -> iajb", self.Y_ia_ri, self.Y_ia_ri) / self.D_iajb
