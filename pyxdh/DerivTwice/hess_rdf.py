# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from warnings import warn
from functools import partial
# pyscf utilities
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceDFSCF, HessSCF
from pyxdh.Utilities import cached_property

# additional
einsum = partial(einsum, optimize="greedy")


class HessDFSCF(DerivTwiceDFSCF, HessSCF):

    @staticmethod
    def _gen_int2c2e_2(aux):
        int2c2e_ipip1 = aux.intor("int2c2e_ipip1").reshape((3, 3, aux.nao, aux.nao))
        int2c2e_ip1ip2 = aux.intor("int2c2e_ip1ip2").reshape((3, 3, aux.nao, aux.nao))
        int2c2e_2 = np.zeros((aux.natm, 3, aux.natm, 3, aux.nao, aux.nao))
        for A, (_, _, A0aux, A1aux) in enumerate(aux.aoslice_by_atom()):
            sAaux = slice(A0aux, A1aux)
            int2c2e_2[A, :, A, :, sAaux, :] += int2c2e_ipip1[:, :, sAaux, :]
            for B, (_, _, B0aux, B1aux) in enumerate(aux.aoslice_by_atom()):
                sBaux = slice(B0aux, B1aux)
                int2c2e_2[A, :, B, :, sAaux, sBaux] += int2c2e_ip1ip2[:, :, sAaux, sBaux]
        int2c2e_2 += int2c2e_2.swapaxes(-1, -2)
        int2c2e_2.shape = (aux.natm * 3, aux.natm * 3, aux.nao, aux.nao)
        return int2c2e_2

    @staticmethod
    def _gen_int3c2e_2(mol, aux):
        ipip1 = int3c_wrapper(mol, aux, "int3c2e_ipip1", "s1")().reshape((3, 3, mol.nao, mol.nao, aux.nao))
        ipvip1 = int3c_wrapper(mol, aux, "int3c2e_ipvip1", "s1")().reshape((3, 3, mol.nao, mol.nao, aux.nao))
        ip1ip2 = int3c_wrapper(mol, aux, "int3c2e_ip1ip2", "s1")().reshape((3, 3, mol.nao, mol.nao, aux.nao))
        ipip2 = int3c_wrapper(mol, aux, "int3c2e_ipip2", "s1")().reshape((3, 3, mol.nao, mol.nao, aux.nao))
        int3c2e_2 = np.zeros((mol.natm, 3, mol.natm, 3, mol.nao, mol.nao, aux.nao))
        for A in range(mol.natm):
            _, _, A0, A1 = mol.aoslice_by_atom()[A]
            _, _, A0aux, A1aux = aux.aoslice_by_atom()[A]
            sA, sAaux = slice(A0, A1), slice(A0aux, A1aux)
            int3c2e_2[A, :, A, :, sA, :, :] += ipip1[:, :, sA, :, :]
            int3c2e_2[A, :, A, :, :, :, sAaux] += 0.5 * ipip2[:, :, :, :, sAaux]
            for B in range(mol.natm):
                _, _, B0, B1 = mol.aoslice_by_atom()[B]
                _, _, B0aux, B1aux = aux.aoslice_by_atom()[B]
                sB, sBaux = slice(B0, B1), slice(B0aux, B1aux)
                int3c2e_2[A, :, B, :, sA, sB, :] += ipvip1[:, :, sA, sB, :]
                int3c2e_2[A, :, B, :, sA, :, sBaux] += ip1ip2[:, :, sA, :, sBaux]
                int3c2e_2[A, :, B, :, sB, :, sAaux] += ip1ip2[:, :, sB, :, sAaux].swapaxes(0, 1)
        int3c2e_2 += int3c2e_2.swapaxes(-2, -3)
        int3c2e_2.shape = (mol.natm * 3, mol.natm * 3, mol.nao, mol.nao, aux.nao)
        return int3c2e_2

    @cached_property
    def eri2_ao(self):
        warn("eri2 should not called in density fitting module!", FutureWarning)
        A, B = self.A, self.B
        return (
            + einsum("AμνP, BκλP, κλ -> ABμνκλ", A.Y_ao_1_jk, B.Y_ao_1_jk)
            + einsum("BμνP, AκλP, κλ -> ABμνκλ", B.Y_ao_1_jk, A.Y_ao_1_jk)
            + einsum("μνP, ABκλP, κλ -> ABμνκλ", A.Y_ao_jk, self.Y_ao_2_jk)
            + einsum("ABμνP, κλP, κλ -> ABμνκλ", self.Y_ao_2_jk, A.Y_ao_jk)
        )

    @cached_property
    def F_2_ao_JKcontrib(self):
        D = self.D
        A, B = self.A, self.B
        j = (
            + einsum("AμνP, BκλP, κλ -> ABμν", A.Y_ao_1_jk, B.Y_ao_1_jk, D)
            + einsum("BμνP, AκλP, κλ -> ABμν", B.Y_ao_1_jk, A.Y_ao_1_jk, D)
            + einsum("μνP, ABκλP, κλ -> ABμν", A.Y_ao_jk, self.Y_ao_2_jk, D)
            + einsum("ABμνP, κλP, κλ -> ABμν", self.Y_ao_2_jk, A.Y_ao_jk, D)
        )
        k = (
            + einsum("AμκP, BνλP, κλ -> ABμν", A.Y_ao_1_jk, B.Y_ao_1_jk, D)
            + einsum("BμκP, AνλP, κλ -> ABμν", B.Y_ao_1_jk, A.Y_ao_1_jk, D)
            + einsum("μκP, ABνλP, κλ -> ABμν", A.Y_ao_jk, self.Y_ao_2_jk, D)
            + einsum("ABμκP, νλP, κλ -> ABμν", self.Y_ao_2_jk, A.Y_ao_jk, D)
        )
        return j, k
