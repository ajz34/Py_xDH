# python utilities
import numpy as np
from opt_einsum import contract as einsum
from warnings import warn
# pyscf utilities
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceDFSCF, DerivOnceDFMP2, GradSCF, GradMP2
from pyxdh.Utilities import cached_property


class GradDFSCF(DerivOnceDFSCF, GradSCF):

    # region Auxiliary Basis Integral Generation

    @staticmethod
    def _gen_int2c2e_1(aux):
        int2c2e_ip1 = aux.intor("int2c2e_ip1")
        int2c2e_1 = np.zeros((aux.natm, 3, aux.nao, aux.nao))
        for A, (_, _, A0aux, A1aux) in enumerate(aux.aoslice_by_atom()):
            sAaux = slice(A0aux, A1aux)
            int2c2e_1[A, :, sAaux, :] -= int2c2e_ip1[:, sAaux, :]
            int2c2e_1[A, :, :, sAaux] -= int2c2e_ip1[:, sAaux, :].swapaxes(-1, -2)
        return int2c2e_1.reshape((aux.natm * 3, aux.nao, aux.nao))

    @staticmethod
    def _gen_int3c2e_1(mol, aux):
        int3c2e_ip1 = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")()
        int3c2e_ip2 = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")()
        int3c2e_1 = np.zeros((mol.natm, 3, mol.nao, mol.nao, aux.nao))
        for A in range(mol.natm):
            _, _, A0, A1 = mol.aoslice_by_atom()[A]
            _, _, A0aux, A1aux = aux.aoslice_by_atom()[A]
            sA, sAaux = slice(A0, A1), slice(A0aux, A1aux)
            int3c2e_1[A, :, sA, :, :] -= int3c2e_ip1[:, sA, :, :]
            int3c2e_1[A, :, :, sA, :] -= int3c2e_ip1[:, sA, :, :].swapaxes(-2, -3)
            int3c2e_1[A, :, :, :, sAaux] -= int3c2e_ip2[:, :, :, sAaux]
        return int3c2e_1.reshape((mol.natm * 3, mol.nao, mol.nao, aux.nao))

    # endregion Auxiliary Basis Integral Generation

    @cached_property
    def eri1_ao(self):
        warn("eri1 should not be called in density fitting module!", FutureWarning)
        return (
            + einsum("AμνP, κλP -> Aμνκλ", self.Y_ao_1_jk, self.Y_ao_jk)
            + einsum("μνP, AκλP -> Aμνκλ", self.Y_ao_jk, self.Y_ao_1_jk))

    def _get_E_1(self):
        so = self.so
        return (
            + einsum("Auv, uv -> A", self.H_1_ao, self.D)
            + 1.0 * einsum("AuvP, klP, uv, kl -> A", self.Y_ao_1_jk, self.Y_ao_jk, self.D, self.D)
            - 0.5 * einsum("AuvP, klP, uk, vl -> A", self.Y_ao_1_jk, self.Y_ao_jk, self.D, self.D)
            - 2 * einsum("ij, Aij -> A", self.F_0_mo[so, so], self.S_1_mo[:, so, so])
        ).reshape((self.mol.natm, 3)) + self.scf_grad.grad_nuc()


class GradDFMP2(DerivOnceDFMP2, GradMP2):
    pass




