import numpy as np
from pyscf.scf import cphf
from abc import ABC, abstractmethod
from functools import partial
import warnings
import os

from pyxdh.DerivOnce import DerivOnceUSCF, DerivOnceUNCDFT
from pyxdh.DerivTwice import DerivTwiceSCF

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivTwiceUSCF(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceUSCF, self).__init__(config)
        self.A = config["deriv_A"]  # type: DerivOnceUSCF
        self.B = config["deriv_B"]  # type: DerivOnceUSCF

    def _get_F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    def _get_S_2_mo(self):
        if not isinstance(self.S_2_ao, np.ndarray):
            return 0
        return np.einsum("ABuv, xup, xvq -> xABpq", self.S_2_ao, self.C, self.C)

    def _get_F_2_mo(self):
        return np.einsum("xABuv, xup, xvq -> xABpq", self.F_2_ao, self.C, self.C)

    def _get_Xi_2(self):
        A = self.A
        B = self.B

        Xi_2 = (
            self.S_2_mo
            + np.einsum("xApm, xBqm -> xABpq", A.U_1, B.U_1)
            + np.einsum("xBpm, xAqm -> xABpq", B.U_1, A.U_1)
        )
        if isinstance(A.S_1_mo, np.ndarray) and isinstance(B.S_1_mo, np.ndarray):
            Xi_2 -= np.einsum("xApm, xBqm -> xABpq", A.S_1_mo, B.S_1_mo)
            Xi_2 -= np.einsum("xBpm, xAqm -> xABpq", B.S_1_mo, A.S_1_mo)
        return Xi_2

    def _get_E_2_U(self):
        A, B = self.A, self.B
        Xi_2 = self.Xi_2
        so, sa = self.so, self.sa
        e, eo = self.e, self.eo
        Ax0_Core = self.A.Ax0_Core

        prop_dim = A.U_1.shape[1]
        E_2_U = np.zeros((prop_dim, prop_dim))
        Ax0_BU = Ax0_Core(sa, so, sa, so)((B.U_1[0, :, :, so[0]], B.U_1[1, :, :, so[1]]))
        for x in range(2):
            E_2_U -= 1 * np.einsum("ABi, i -> AB", Xi_2[x].diagonal(0, -1, -2)[:, :, so[x]], eo[x])
            E_2_U += 2 * np.einsum("Bpi, Api -> AB", B.U_1[x][:, :, so[x]], A.F_1_mo[x][:, :, so[x]])
            E_2_U += 2 * np.einsum("Api, Bpi -> AB", A.U_1[x][:, :, so[x]], B.F_1_mo[x][:, :, so[x]])
            E_2_U += 2 * np.einsum("Api, Bpi, p -> AB", A.U_1[x][:, :, so[x]], B.U_1[x][:, :, so[x]], e[x])
            E_2_U += 2 * np.einsum("Api, Bpi -> AB", A.U_1[x][:, :, so[x]], Ax0_BU[x])

        return E_2_U

    def _get_pdB_F_A_mo(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        pdB_F_A_mo = (
            + self.F_2_mo
            + np.einsum("xApm, xBmq -> xABpq", A.F_1_mo, B.U_1)
            + np.einsum("xAmq, xBmp -> xABpq", A.F_1_mo, B.U_1)
            + np.array(A.Ax1_Core(sa, sa, sa, so)((B.U_1[0][:, :, so[0]], B.U_1[1][:, :, so[1]])))
        )
        return pdB_F_A_mo

    def _get_pdB_S_A_mo(self):
        A, B = self.A, self.B
        pdB_S_A_mo = (
            + self.S_2_mo
            + np.einsum("xApm, xBmq -> xABpq", A.S_1_mo, B.U_1)
            + np.einsum("xAmq, xBmp -> xABpq", A.S_1_mo, B.U_1)
        )
        return pdB_S_A_mo

    def _get_pdB_B_A(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        Ax0_Core = A.Ax0_Core
        pdB_B_A = (
            + self.pdB_F_A_mo
            - np.einsum("xABpq, xq -> xABpq", self.pdB_S_A_mo, self.e)
            - np.einsum("xApm, xBqm -> xABpq", A.S_1_mo, B.pdA_F_0_mo)
            - 0.5 * np.array(B.Ax1_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))).swapaxes(1, 2)
            - 0.5 * np.array(Ax0_Core(sa, sa, so, so)((self.pdB_S_A_mo[0][:, :, so[0], so[0]], self.pdB_S_A_mo[1][:, :, so[1], so[1]])))
            - np.array(Ax0_Core(sa, sa, sa, so)((
                    np.einsum("Bml, Akl -> ABmk", B.U_1[0][:, :, so[0]], A.S_1_mo[0][:, so[0], so[0]]),
                    np.einsum("Bml, Akl -> ABmk", B.U_1[1][:, :, so[1]], A.S_1_mo[1][:, so[1], so[1]]),
                )))
            - 0.5 * np.einsum("xBmp, xAmq -> xABpq", B.U_1, np.array(Ax0_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))))
            - 0.5 * np.einsum("xBmq, xAmp -> xABpq", B.U_1, np.array(Ax0_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))))
        )
        return pdB_B_A
