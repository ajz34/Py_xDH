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

    def _get_S_2_mo(self):
        if not isinstance(self.S_2_ao, np.ndarray):
            return 0
        return np.einsum("ABuv, xup, xvq -> xABpq", self.S_2_ao, self.C, self.C)

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
