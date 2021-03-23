# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceUSCF, DerivOnceUMP2
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceMP2
from pyxdh.Utilities import cached_property


class DerivTwiceUSCF(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceUSCF, self).__init__(config)
        self.A = self.A  # type: DerivOnceUSCF
        self.B = self.B  # type: DerivOnceUSCF

    @cached_property
    def F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    @cached_property
    def S_2_mo(self):
        if not isinstance(self.S_2_ao, np.ndarray):
            return 0
        return einsum("ABuv, xup, xvq -> xABpq", self.S_2_ao, self.C, self.C)

    @cached_property
    def F_2_mo(self):
        if not isinstance(self.F_2_ao, np.ndarray):
            return 0
        return einsum("xABuv, xup, xvq -> xABpq", self.F_2_ao, self.C, self.C)

    @cached_property
    def Xi_2(self):
        A = self.A
        B = self.B

        Xi_2 = (
            self.S_2_mo
            + einsum("xApm, xBqm -> xABpq", A.U_1, B.U_1)
            + einsum("xBpm, xAqm -> xABpq", B.U_1, A.U_1)
        )
        if isinstance(A.S_1_mo, np.ndarray) and isinstance(B.S_1_mo, np.ndarray):
            Xi_2 -= einsum("xApm, xBqm -> xABpq", A.S_1_mo, B.S_1_mo)
            Xi_2 -= einsum("xBpm, xAqm -> xABpq", B.S_1_mo, A.S_1_mo)
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
            E_2_U -= 1 * einsum("ABi, i -> AB", Xi_2[x].diagonal(0, -1, -2)[:, :, so[x]], eo[x])
            E_2_U += 2 * einsum("Bpi, Api -> AB", B.U_1[x][:, :, so[x]], A.F_1_mo[x][:, :, so[x]])
            E_2_U += 2 * einsum("Api, Bpi -> AB", A.U_1[x][:, :, so[x]], B.F_1_mo[x][:, :, so[x]])
            E_2_U += 2 * einsum("Api, Bpi, p -> AB", A.U_1[x][:, :, so[x]], B.U_1[x][:, :, so[x]], e[x])
            E_2_U += 2 * einsum("Api, Bpi -> AB", A.U_1[x][:, :, so[x]], Ax0_BU[x])

        return E_2_U

    @cached_property
    def pdB_F_A_mo(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        pdB_F_A_mo = (
            + self.F_2_mo
            + einsum("xApm, xBmq -> xABpq", A.F_1_mo, B.U_1)
            + einsum("xAmq, xBmp -> xABpq", A.F_1_mo, B.U_1)
            + np.array(A.Ax1_Core(sa, sa, sa, so)((B.U_1[0][:, :, so[0]], B.U_1[1][:, :, so[1]])))
        )
        return pdB_F_A_mo

    @cached_property
    def pdB_S_A_mo(self):
        A, B = self.A, self.B
        if not isinstance(A.S_1_mo, np.ndarray):
            return self.S_2_mo
        pdB_S_A_mo = (
            + self.S_2_mo
            + einsum("xApm, xBmq -> xABpq", A.S_1_mo, B.U_1)
            + einsum("xAmq, xBmp -> xABpq", A.S_1_mo, B.U_1)
        )
        return pdB_S_A_mo

    @cached_property
    def pdB_B_A(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        Ax0_Core = A.Ax0_Core
        pdB_B_A = (
            + self.pdB_F_A_mo
            - einsum("xABpq, xq -> xABpq", self.pdB_S_A_mo, self.e)
            - einsum("xApm, xBqm -> xABpq", A.S_1_mo, B.pdA_F_0_mo)
            - 0.5 * np.array(B.Ax1_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))).swapaxes(1, 2)
            - 0.5 * np.array(Ax0_Core(sa, sa, so, so)((self.pdB_S_A_mo[0][:, :, so[0], so[0]], self.pdB_S_A_mo[1][:, :, so[1], so[1]])))
            - np.array(Ax0_Core(sa, sa, sa, so)((
                    einsum("Bml, Akl -> ABmk", B.U_1[0][:, :, so[0]], A.S_1_mo[0][:, so[0], so[0]]),
                    einsum("Bml, Akl -> ABmk", B.U_1[1][:, :, so[1]], A.S_1_mo[1][:, so[1], so[1]]),
                )))
            - 0.5 * einsum("xBmp, xAmq -> xABpq", B.U_1, np.array(Ax0_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))))
            - 0.5 * einsum("xBmq, xAmp -> xABpq", B.U_1, np.array(Ax0_Core(sa, sa, so, so)((A.S_1_mo[0][:, so[0], so[0]], A.S_1_mo[1][:, so[1], so[1]]))))
        )
        return pdB_B_A


class DerivTwiceUMP2(DerivTwiceMP2, DerivTwiceUSCF, DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceUMP2, self).__init__(config)
        self.A = self.A  # type: DerivOnceUMP2
        self.B = self.B  # type: DerivOnceUMP2

    @cached_property
    def pdB_pdpA_eri0_iajb(self):
        A, B = self.A, self.B
        so, sv, sa = self.so, self.sv, self.sa
        eri1_mo, U_1 = A.eri1_mo, B.U_1
        sigma_list = [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 1],
        ]
        pdB_pdpA_eri0_iajb = [None, None, None]
        for x, y, z in sigma_list:
            pdB_pdpA_eri0_iajb[x] = (
                + np.einsum("ABuvkl, up, vq, kr, ls -> ABpqrs", self.eri2_ao, self.Co[y], self.Cv[y], self.Co[z], self.Cv[z])
                + np.einsum("Apjkl, Bpi -> ABijkl", eri1_mo[x][:, sa[y], sv[y], so[z], sv[z]], U_1[y][:, :, so[y]])
                + np.einsum("Aipkl, Bpj -> ABijkl", eri1_mo[x][:, so[y], sa[y], so[z], sv[z]], U_1[y][:, :, sv[y]])
                + np.einsum("Aijpl, Bpk -> ABijkl", eri1_mo[x][:, so[y], sv[y], sa[z], sv[z]], U_1[z][:, :, so[z]])
                + np.einsum("Aijkp, Bpl -> ABijkl", eri1_mo[x][:, so[y], sv[y], so[z], sa[z]], U_1[z][:, :, sv[z]])
            )
        return pdB_pdpA_eri0_iajb

    def _get_RHS_B(self):
        B = self.B
        so, sv, sa = self.so, self.sv, self.sa
        U_1, D_r, pdB_F_0_mo, eri0_mo = B.U_1, B.D_r, B.pdA_F_0_mo, B.eri0_mo
        Ax0_Core, Ax1_Core = B.Ax0_Core, B.Ax1_Core
        pdB_D_r_oovv = B.pdA_D_r_oovv
        prop_dim = pdB_D_r_oovv[0].shape[0]

        # total partial eri0
        pdA_eri0_mo = B.pdA_eri0_mo

        RHS_B = [np.zeros((prop_dim, self.nvir[0], self.nocc[0])), np.zeros((prop_dim, self.nvir[1], self.nocc[1]))]

        # D_r Part
        Ax0_pdB_D_r_oovv = Ax0_Core(sv, so, sa, sa)(pdB_D_r_oovv)
        Ax1_D_r = Ax1_Core(sv, so, sa, sa)(D_r)
        Ax0_D_r = Ax0_Core(sa, sa, sa, sa)(D_r)
        Ax0_U_D = Ax0_Core(sv, so, sa, sa)((
            np.einsum("Amp, pq -> Amq", U_1[0], D_r[0]) + np.einsum("Amq, pq -> Apm", U_1[0], D_r[0]),
            np.einsum("Amp, pq -> Amq", U_1[1], D_r[1]) + np.einsum("Amq, pq -> Apm", U_1[1], D_r[1]),
        ))
        for x in range(2):
            RHS_B[x] = Ax0_pdB_D_r_oovv[x]
            if isinstance(Ax1_D_r, np.ndarray):
                RHS_B[x] += Ax1_D_r[x]
            RHS_B[x] += np.einsum("Apa, pi -> Aai", U_1[x][:, sa[x], sv[x]], Ax0_D_r[x][sa[x], so[x]])
            RHS_B[x] += np.einsum("Api, ap -> Aai", U_1[x][:, sa[x], so[x]], Ax0_D_r[x][sv[x], sa[x]])
            RHS_B[x] += Ax0_U_D[x]
        # (ea - ei) * Dai
        for x in range(2):
            RHS_B[x] += np.einsum("Aca, ci -> Aai", pdB_F_0_mo[x][:, sv[x], sv[x]], D_r[x][sv[x], so[x]])
            RHS_B[x] -= np.einsum("Aki, ak -> Aai", pdB_F_0_mo[x][:, so[x], so[x]], D_r[x][sv[x], so[x]])
        # 2-pdm part
        RHS_B[0] += (
            - 4 * np.einsum("Ajakb, ijbk -> Aai", B.pdA_T_iajb[0], eri0_mo[0][   so[0], so[0], sv[0], so[0]])
            + 4 * np.einsum("Aibjc, abjc -> Aai", B.pdA_T_iajb[0], eri0_mo[0][   sv[0], sv[0], so[0], sv[0]])
            - 4 * np.einsum("jakb, Aijbk -> Aai", B.T_iajb[0], pdA_eri0_mo[0][:, so[0], so[0], sv[0], so[0]])
            + 4 * np.einsum("ibjc, Aabjc -> Aai", B.T_iajb[0], pdA_eri0_mo[0][:, sv[0], sv[0], so[0], sv[0]]))
        RHS_B[0] += (
            - 2 * np.einsum("Ajakb, ijbk -> Aai", B.pdA_T_iajb[1], eri0_mo[1][   so[0], so[0], sv[1], so[1]])
            + 2 * np.einsum("Aibjc, abjc -> Aai", B.pdA_T_iajb[1], eri0_mo[1][   sv[0], sv[0], so[1], sv[1]])
            - 2 * np.einsum("jakb, Aijbk -> Aai", B.T_iajb[1], pdA_eri0_mo[1][:, so[0], so[0], sv[1], so[1]])
            + 2 * np.einsum("ibjc, Aabjc -> Aai", B.T_iajb[1], pdA_eri0_mo[1][:, sv[0], sv[0], so[1], sv[1]]))
        RHS_B[1] += (
            - 4 * np.einsum("Ajakb, ijbk -> Aai", B.pdA_T_iajb[2], eri0_mo[2][   so[1], so[1], sv[1], so[1]])
            + 4 * np.einsum("Aibjc, abjc -> Aai", B.pdA_T_iajb[2], eri0_mo[2][   sv[1], sv[1], so[1], sv[1]])
            - 4 * np.einsum("jakb, Aijbk -> Aai", B.T_iajb[2], pdA_eri0_mo[2][:, so[1], so[1], sv[1], so[1]])
            + 4 * np.einsum("ibjc, Aabjc -> Aai", B.T_iajb[2], pdA_eri0_mo[2][:, sv[1], sv[1], so[1], sv[1]]))
        RHS_B[1] += (
            - 2 * np.einsum("Akbja, bkij -> Aai", B.pdA_T_iajb[1], eri0_mo[1][   sv[0], so[0], so[1], so[1]])
            + 2 * np.einsum("Ajcib, jcab -> Aai", B.pdA_T_iajb[1], eri0_mo[1][   so[0], sv[0], sv[1], sv[1]])
            - 2 * np.einsum("kbja, Abkij -> Aai", B.T_iajb[1], pdA_eri0_mo[1][:, sv[0], so[0], so[1], so[1]])
            + 2 * np.einsum("jcib, Ajcab -> Aai", B.T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], sv[1], sv[1]]))

        return tuple(RHS_B)

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = np.zeros((A.U_1[0].shape[0], B.U_1[0].shape[0]))
        for x in range(2):
            E_2_MP2_Contrib += (
                # D_r * B
                + np.einsum("pq, ABpq -> AB", self.D_r[x], self.pdB_B_A[x])
                + np.einsum("Bpq, Apq -> AB", B.pdA_D_r_oovv[x], A.B_1[x])
                + np.einsum("Aai, Bai -> AB", A.U_1[x][:, sv[x], so[x]], self.RHS_B[x])
                # W_I * S
                + np.einsum("pq, ABpq -> AB", self.W_I[x], self.pdB_S_A_mo[x])
                + np.einsum("Bpq, Apq -> AB", B.pdA_W_I[x], A.S_1_mo[x])
            )
        # T * g
        sigma_list = [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 1],
        ]
        for x, y, z in sigma_list:
            E_2_MP2_Contrib += (
                + 2 * np.einsum("Biajb, Aiajb -> AB", B.pdA_T_iajb[x], A.eri1_mo[x][:, so[y], sv[y], so[z], sv[z]])
                + 2 * np.einsum("iajb, ABiajb -> AB", self.T_iajb[x], self.pdB_pdpA_eri0_iajb[x])
            )
        return E_2_MP2_Contrib
