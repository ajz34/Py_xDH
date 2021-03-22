import numpy as np
from abc import ABC
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceUSCF, DerivTwiceMP2, DerivTwiceNCDFT
from pyxdh.DerivOnce import DerivOnceUMP2, DerivOnceXDH

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivTwiceUMP2(DerivTwiceMP2, DerivTwiceUSCF, DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceUMP2, self).__init__(config)
        self.A = config["deriv_A"]  # type: DerivOnceUMP2
        self.B = config["deriv_B"]  # type: DerivOnceUMP2

    def _get_pdB_D_r_oovv(self):
        B = self.B
        so, sv = self.so, self.sv
        nmo = self.nmo
        T_iajb, t_iajb = B.T_iajb, B.t_iajb
        pdA_t_iajb, pdA_T_iajb = B.pdA_t_iajb, B.pdA_T_iajb

        pdB_D_r_oovv = np.zeros((2, pdA_t_iajb[0].shape[0], nmo, nmo))
        pdB_D_r_oovv[0, :, so[0], so[0]] = (
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[0], pdA_t_iajb[0])
                - np.einsum("iakb, Ajakb -> Aij", T_iajb[1], pdA_t_iajb[1])
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[0], t_iajb[0])
                - np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[1], t_iajb[1]))
        pdB_D_r_oovv[1, :, so[1], so[1]] = (
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[2], pdA_t_iajb[2])
                - np.einsum("kbia, Akbja -> Aij", T_iajb[1], pdA_t_iajb[1])
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[2], t_iajb[2])
                - np.einsum("Akbia, kbja -> Aij", pdA_T_iajb[1], t_iajb[1]))
        pdB_D_r_oovv[0, :, sv[0], sv[0]] = (
                + 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[0], pdA_t_iajb[0])
                + np.einsum("iajc, Aibjc -> Aab", T_iajb[1], pdA_t_iajb[1])
                + 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[0], t_iajb[0])
                + np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[1], t_iajb[1]))
        pdB_D_r_oovv[1, :, sv[1], sv[1]] = (
                + 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[2], pdA_t_iajb[2])
                + np.einsum("jcia, Ajcib -> Aab", T_iajb[1], pdA_t_iajb[1])
                + 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[2], t_iajb[2])
                + np.einsum("Ajcia, jcib -> Aab", pdA_T_iajb[1], t_iajb[1]))
        return pdB_D_r_oovv

    def _get_pdB_W_I(self):
        so, sv = self.so, self.sv
        natm, nmo = self.natm, self.nmo
        pdA_T_iajb, T_iajb = self.B.pdA_T_iajb, self.B.T_iajb
        eri0_mo, pdA_eri0_mo = self.B.eri0_mo, self.B.pdA_eri0_mo
        pdB_W_I = np.zeros((2, pdA_T_iajb[0].shape[0], nmo, nmo))
        pdB_W_I[0, :, so[0], so[0]] = (
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[0], eri0_mo[0][so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[0], pdA_eri0_mo[0][:, so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("iakb, Ajakb -> Aij", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        pdB_W_I[1, :, so[1], so[1]] = (
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[2], eri0_mo[2][so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("Akbia, kbja -> Aij", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[2], pdA_eri0_mo[2][:, so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("kbia, Akbja -> Aij", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        # vir-vir part
        pdB_W_I[0, :, sv[0], sv[0]] = (
                - 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[0], eri0_mo[0][so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[0], pdA_eri0_mo[0][:, so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("iajc, Aibjc -> Aab", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        pdB_W_I[1, :, sv[1], sv[1]] = (
                - 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[2], eri0_mo[2][so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("Ajcia, jcib -> Aab", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[2], pdA_eri0_mo[2][:, so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("jcia, Ajcib -> Aab", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        # vir-occ part
        pdB_W_I[0, :, sv[0], so[0]] = (
                - 4 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[0], eri0_mo[0][so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[1], eri0_mo[1][so[0], so[0], sv[1], so[1]])
                - 4 * np.einsum("jakb, Aijbk -> Aai", T_iajb[0], pdA_eri0_mo[0][:, so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("jakb, Aijbk -> Aai", T_iajb[1], pdA_eri0_mo[1][:, so[0], so[0], sv[1], so[1]]))
        pdB_W_I[1, :, sv[1], so[1]] = (
                - 4 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[2], eri0_mo[2][so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("Akbja, bkij -> Aai", pdA_T_iajb[1], eri0_mo[1][sv[0], so[0], so[1], so[1]])
                - 4 * np.einsum("jakb, Aijbk -> Aai", T_iajb[2], pdA_eri0_mo[2][:, so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("kbja, Abkij -> Aai", T_iajb[1], pdA_eri0_mo[1][:, sv[0], so[0], so[1], so[1]]))
        return pdB_W_I

    def _get_pdB_pdpA_eri0_iajb(self):
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

