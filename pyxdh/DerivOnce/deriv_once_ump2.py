import numpy as np
from abc import ABC
from functools import partial
import os

from pyscf.scf import ucphf

from pyxdh.DerivOnce import DerivOnceUSCF, DerivOnceUNCDFT, DerivOnceMP2

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceUMP2(DerivOnceUSCF, DerivOnceMP2, ABC):

    def _get_D_iajb(self):
        eo, ev = self.eo, self.ev
        D_iajb = (
            eo[0][:, None, None, None] - ev[0][None, :, None, None] + eo[0][None, None, :, None] - ev[0][None, None, None, :],
            eo[0][:, None, None, None] - ev[0][None, :, None, None] + eo[1][None, None, :, None] - ev[1][None, None, None, :],
            eo[1][:, None, None, None] - ev[1][None, :, None, None] + eo[1][None, None, :, None] - ev[1][None, None, None, :]
        )
        return D_iajb

    def _get_t_iajb(self):
        so, sv = self.so, self.sv
        eri0_mo = self.eri0_mo
        D_iajb = self.D_iajb
        t_iajb = (
            eri0_mo[0][so[0], sv[0], so[0], sv[0]] / D_iajb[0],
            eri0_mo[1][so[0], sv[0], so[1], sv[1]] / D_iajb[1],
            eri0_mo[2][so[1], sv[1], so[1], sv[1]] / D_iajb[2]
        )
        return t_iajb

    def _get_T_iajb(self):
        t_iajb = self.t_iajb
        cc, ss, os = self.cc, self.ss, self.os
        T_iajb = (
            0.5 * cc * ss * (t_iajb[0] - t_iajb[0].swapaxes(-1, -3)),
            cc * os * t_iajb[1],
            0.5 * cc * ss * (t_iajb[2] - t_iajb[2].swapaxes(-1, -3))
        )
        return T_iajb

    def _get_W_I(self):
        so, sv = self.so, self.sv
        nmo = self.nmo
        t_iajb, T_iajb, D_iajb, eri0_mo = self.t_iajb, self.T_iajb, self.D_iajb, self.eri0_mo
        W_I = np.zeros((2, nmo, nmo))
        # occ-occ part
        W_I[0, so[0], so[0]] = (
                - 2 * np.einsum("iakb, jakb -> ij", T_iajb[0], t_iajb[0] * D_iajb[0])
                - np.einsum("iakb, jakb -> ij", T_iajb[1], t_iajb[1] * D_iajb[1]))
        W_I[1, so[1], so[1]] = (
                - 2 * np.einsum("iakb, jakb -> ij", T_iajb[2], t_iajb[2] * D_iajb[2])
                - np.einsum("kbia, kbja -> ij", T_iajb[1], t_iajb[1] * D_iajb[1]))
        # vir-vir part
        W_I[0, sv[0], sv[0]] = (
                - 2 * np.einsum("iajc, ibjc -> ab", T_iajb[0], t_iajb[0] * D_iajb[0])
                - np.einsum("iajc, ibjc -> ab", T_iajb[1], t_iajb[1] * D_iajb[1]))
        W_I[1, sv[1], sv[1]] = (
                - 2 * np.einsum("iajc, ibjc -> ab", T_iajb[2], t_iajb[2] * D_iajb[2])
                - np.einsum("jcia, jcib -> ab", T_iajb[1], t_iajb[1] * D_iajb[1]))
        # vir-occ part
        W_I[0, sv[0], so[0]] = (
                - 4 * np.einsum("jakb, ijbk -> ai", T_iajb[0], eri0_mo[0][so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("jakb, ijbk -> ai", T_iajb[1], eri0_mo[1][so[0], so[0], sv[1], so[1]]))
        W_I[1, sv[1], so[1]] = (
                - 4 * np.einsum("jakb, ijbk -> ai", T_iajb[2], eri0_mo[2][so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("kbja, bkij -> ai", T_iajb[1], eri0_mo[1][sv[0], so[0], so[1], so[1]]))
        return W_I

    def _get_L(self):
        nmo = self.nmo
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        eri0_mo = self.eri0_mo
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        D_r_oovv = np.zeros((2, nmo, nmo))
        if self._D_r is NotImplemented:
            D_r_oovv[0, so[0], so[0]] = (
                    - 2 * np.einsum("iakb, jakb -> ij", T_iajb[0], t_iajb[0])
                    - np.einsum("iakb, jakb -> ij", T_iajb[1], t_iajb[1]))
            D_r_oovv[1, so[1], so[1]] = (
                    - 2 * np.einsum("iakb, jakb -> ij", T_iajb[2], t_iajb[2])
                    - np.einsum("kbia, kbja -> ij", T_iajb[1], t_iajb[1]))
            D_r_oovv[0, sv[0], sv[0]] = (
                    + 2 * np.einsum("iajc, ibjc -> ab", T_iajb[0], t_iajb[0])
                    + np.einsum("iajc, ibjc -> ab", T_iajb[1], t_iajb[1]))
            D_r_oovv[1, sv[1], sv[1]] = (
                    + 2 * np.einsum("iajc, ibjc -> ab", T_iajb[2], t_iajb[2])
                    + np.einsum("jcia, jcib -> ab", T_iajb[1], t_iajb[1]))
            self._D_r = D_r_oovv
        else:
            D_r_oovv[0, so[0], so[0]] = self.D_r[0, so[0], so[0]]
            D_r_oovv[1, so[1], so[1]] = self.D_r[1, so[1], so[1]]
            D_r_oovv[0, sv[0], sv[0]] = self.D_r[0, sv[0], sv[0]]
            D_r_oovv[1, sv[1], sv[1]] = self.D_r[1, sv[1], sv[1]]

        L = Ax0_Core(sv, so, sa, sa)(D_r_oovv)
        L[0][:] += (
                - 4 * np.einsum("jakb, ijbk -> ai", T_iajb[0], eri0_mo[0][so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("jakb, ijbk -> ai", T_iajb[1], eri0_mo[1][so[0], so[0], sv[1], so[1]])
                + 4 * np.einsum("ibjc, abjc -> ai", T_iajb[0], eri0_mo[0][sv[0], sv[0], so[0], sv[0]])
                + 2 * np.einsum("ibjc, abjc -> ai", T_iajb[1], eri0_mo[1][sv[0], sv[0], so[1], sv[1]])
        )
        L[1][:] += (
                - 4 * np.einsum("jakb, ijbk -> ai", T_iajb[2], eri0_mo[2][so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("kbja, bkij -> ai", T_iajb[1], eri0_mo[1][sv[0], so[0], so[1], so[1]])
                + 4 * np.einsum("ibjc, abjc -> ai", T_iajb[2], eri0_mo[2][sv[1], sv[1], so[1], sv[1]])
                + 2 * np.einsum("jcib, jcab -> ai", T_iajb[1], eri0_mo[1][so[0], sv[0], sv[1], sv[1]])
        )
        return L

    def _get_eng(self):
        T_iajb, t_iajb, D_iajb = self.T_iajb, self.t_iajb, self.D_iajb
        eng = DerivOnceUSCF._get_eng(self)
        eng += np.array([(T_iajb[i] * t_iajb[i] * D_iajb[i]).sum() for i in range(3)]).sum()
        return eng

    def _get_D_r(self):
        L = self.L
        D_r = self._D_r
        so, sv = self.so, self.sv

        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo

        def fx(X):
            X_alpha = X[:, :nocc[0] * nvir[0]].reshape((nvir[0], nocc[0]))
            X_beta = X[:, nocc[0] * nvir[0]:].reshape((nvir[1], nocc[1]))
            Ax = Ax0_Core(sv, so, sv, so, in_cphf=True)((X_alpha, X_beta))
            result = np.concatenate([Ax[0].reshape(-1), Ax[1].reshape(-1)])
            return result

        D_r_vo = ucphf.solve(fx, e, mo_occ, L, max_cycle=100, tol=self.cphf_tol)[0]
        D_r[0][sv[0], so[0]] = D_r_vo[0]
        D_r[1][sv[1], so[1]] = D_r_vo[1]
        return D_r

    def _get_pdA_eri0_mo(self):
        eri0_mo = self.eri0_mo
        U_1 = self.U_1
        nmo = self.nmo
        pdA_eri0_mo = np.zeros((3, self.H_1_ao.shape[0], nmo, nmo, nmo, nmo))
        pdA_eri0_mo += self.eri1_mo
        sigma_list = [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 1],
        ]
        for x, y, z in sigma_list:
            pdA_eri0_mo[x] += (
                + np.einsum("pjkl, Api -> Aijkl", eri0_mo[x], U_1[y])
                + np.einsum("ipkl, Apj -> Aijkl", eri0_mo[x], U_1[y])
                + np.einsum("ijpl, Apk -> Aijkl", eri0_mo[x], U_1[z])
                + np.einsum("ijkp, Apl -> Aijkl", eri0_mo[x], U_1[z])
            )
        return pdA_eri0_mo

    def _get_pdA_t_iajb(self):
        so, sv = self.so, self.sv
        D_iajb = self.D_iajb
        pdA_F_0_mo = self.pdA_F_0_mo
        t_iajb = self.t_iajb
        pdA_eri0_mo = self.pdA_eri0_mo

        sigma_list = [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 1],
        ]
        pdA_t_iajb = [np.copy(pdA_eri0_mo[x, :, so[y], sv[y], so[z], sv[z]]) for x, y, z in sigma_list]
        for x, y, z, in sigma_list:
            pdA_t_iajb[x] += (
                - np.einsum("Aki, kajb -> Aiajb", pdA_F_0_mo[y][:, so[y], so[y]], t_iajb[x])
                - np.einsum("Akj, iakb -> Aiajb", pdA_F_0_mo[z][:, so[z], so[z]], t_iajb[x])
                + np.einsum("Aca, icjb -> Aiajb", pdA_F_0_mo[y][:, sv[y], sv[y]], t_iajb[x])
                + np.einsum("Acb, iajc -> Aiajb", pdA_F_0_mo[z][:, sv[z], sv[z]], t_iajb[x])
            )
            pdA_t_iajb[x] /= D_iajb[x]
        return tuple(pdA_t_iajb)

    def _get_pdA_T_iajb(self):
        pdA_t_iajb = self.pdA_t_iajb
        cc, ss, os = self.cc, self.ss, self.os
        pdA_T_iajb = (
            0.5 * cc * ss * (pdA_t_iajb[0] - pdA_t_iajb[0].swapaxes(-1, -3)),
            cc * os * pdA_t_iajb[1],
            0.5 * cc * ss * (pdA_t_iajb[2] - pdA_t_iajb[2].swapaxes(-1, -3))
        )
        return pdA_T_iajb


class DerivOnceUXDH(DerivOnceUMP2, DerivOnceUNCDFT, ABC):

    def _get_L(self):
        sv, so = self.sv, self.so
        L = super(DerivOnceUXDH, self)._get_L()
        L[0][:] += 2 * self.nc_deriv.F_0_mo[0][sv[0], so[0]]
        L[1][:] += 2 * self.nc_deriv.F_0_mo[1][sv[1], so[1]]
        return L

    def _get_eng(self):
        T_iajb, t_iajb, D_iajb = self.T_iajb, self.t_iajb, self.D_iajb
        eng = DerivOnceUNCDFT._get_eng(self)
        eng += np.array([(T_iajb[i] * t_iajb[i] * D_iajb[i]).sum() for i in range(3)]).sum()
        return eng
