# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC
import warnings
from typing import Tuple, Callable
# pyscf utilities
from pyscf.scf._response_functions import _gen_uhf_response
from pyscf import dft, scf, lib, hessian
from pyscf.scf import ucphf
# pyxdh utilities
from pyxdh.DerivOnce.deriv_once_r import DerivOnceSCF, DerivOnceNCDFT, DerivOnceMP2
from pyxdh.Utilities import timing, cached_property
# additional definition for hessian
scf.uhf.UHF.Hessian = lib.class_as_method(hessian.uhf.Hessian)
dft.uks.UKS.Hessian = lib.class_as_method(hessian.uks.Hessian)


class DerivOnceUSCF(DerivOnceSCF, ABC):

    def __init__(self, config):
        self._nocc = NotImplemented  # type: Tuple[int, int]
        super(DerivOnceUSCF, self).__init__(config)

    def initialization_scf(self):
        if self.init_scf:
            self.mo_occ = self.scf_eng.mo_occ
            self.C = self.scf_eng.mo_coeff
            self.e = self.scf_eng.mo_energy
            self.nocc = self.mol.nelec
        return

    @property
    def nvir(self) -> Tuple[int, int]:
        return self.nmo - self.nocc[0], self.nmo - self.nocc[1]

    @property
    def so(self) -> Tuple[slice, slice]:
        return slice(0, self.nocc[0]), slice(0, self.nocc[1])

    @property
    def sv(self) -> Tuple[slice, slice]:
        return slice(self.nocc[0], self.nmo), slice(self.nocc[1], self.nmo)

    @property
    def sa(self) -> Tuple[slice, slice]:
        return slice(0, self.nmo), slice(0, self.nmo)

    @property
    def Co(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.C[0, :, self.so[0]], self.C[1, :, self.so[1]]

    @property
    def Cv(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.C[0, :, self.sv[0]], self.C[1, :, self.sv[1]]

    @property
    def eo(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.e[0, self.so[0]], self.e[1, self.so[1]]

    @property
    def ev(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.e[0, self.sv[0]], self.e[1, self.sv[1]]

    @property
    def U_1_vo(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.U_1[0, :, self.sv[0], self.so[0]], self.U_1[1, :, self.sv[1], self.so[1]]

    @property
    def U_1_ov(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.U_1[0, :, self.so[0], self.sv[0]], self.U_1[1, :, self.so[1], self.sv[1]]

    @cached_property
    def D(self) -> np.ndarray:
        return einsum("xup, xp, xvp -> xuv", self.C, self.occ, self.C)

    @cached_property
    def H_0_mo(self) -> np.ndarray:
        return einsum("xup, uv, xvq -> xpq", self.C, self.H_0_ao, self.C)

    @cached_property
    def S_0_mo(self) -> np.ndarray:
        return einsum("xup, uv, xvq -> xpq", self.C, self.S_0_ao, self.C)

    @cached_property
    def eri0_mo(self) -> np.ndarray:
        nmo = self.nmo
        C = self.C
        eri0_ao = self.eri0_ao
        eri0_mo = np.zeros((3, nmo, nmo, nmo, nmo))
        eri0_mo[0] = einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[0], C[0], C[0], C[0])
        eri0_mo[1] = einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[0], C[0], C[1], C[1])
        eri0_mo[2] = einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[1], C[1], C[1], C[1])
        return eri0_mo

    @cached_property
    def F_0_mo(self) -> np.ndarray:
        return einsum("xup, xuv, xvq -> xpq", self.C, self.F_0_ao, self.C)

    @cached_property
    def H_1_mo(self) -> np.ndarray:
        if not isinstance(self.H_1_ao, np.ndarray):
            return 0
        return einsum("Auv, xup, xvq -> xApq", self.H_1_ao, self.C, self.C)

    @cached_property
    def S_1_mo(self) -> np.ndarray:
        if not isinstance(self.S_1_ao, np.ndarray):
            return 0
        return einsum("Auv, xup, xvq -> xApq", self.S_1_ao, self.C, self.C)

    @cached_property
    def eri1_mo(self) -> np.ndarray or int:
        if not isinstance(self.eri1_ao, np.ndarray):
            return 0
        nmo = self.nmo
        C = self.C
        eri1_ao = self.eri1_ao
        eri1_mo = np.zeros((3, eri1_ao.shape[0], nmo, nmo, nmo, nmo))
        eri1_mo[0] = einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[0], C[0], C[0], C[0])
        eri1_mo[1] = einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[0], C[0], C[1], C[1])
        eri1_mo[2] = einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[1], C[1], C[1], C[1])
        return eri1_mo

    @cached_property
    def F_1_mo(self) -> np.ndarray or int:
        if not isinstance(self.F_1_ao, np.ndarray):
            return 0
        return einsum("xAuv, xup, xvq -> xApq", self.F_1_ao, self.C, self.C)

    @cached_property
    def B_1(self) -> np.ndarray:
        sa = self.sa
        so = self.so

        B_1 = np.copy(self.F_1_mo)
        if isinstance(self.S_1_mo, np.ndarray):
            B_1 += (
                - einsum("xApq, xq -> xApq", self.S_1_mo, self.e)
                - 0.5 * np.array(self.Ax0_Core(sa, sa, so, so)((self.S_1_mo[0, :, so[0], so[0]], self.S_1_mo[1, :, so[1], so[1]])))
            )
        return B_1

    @cached_property
    def pdA_F_0_mo(self) -> np.ndarray:
        F_1_mo = self.F_1_mo
        U_1 = self.U_1
        e = self.e
        Ax0_Core = self.Ax0_Core
        so, sa = self.so, self.sa

        pdA_F_0_mo = (
            + F_1_mo
            + einsum("xApq, xp -> xApq", U_1, e)
            + einsum("xAqp, xq -> xApq", U_1, e)
            + Ax0_Core(sa, sa, sa, so)((U_1[0, :, :, so[0]], U_1[1, :, :, so[1]]))
        )
        return pdA_F_0_mo

    def Ax0_Core(self,
                 si: Tuple[slice, slice], sj: Tuple[slice, slice],
                 sk: Tuple[slice, slice], sl: Tuple[slice, slice],
                 reshape=True, in_cphf=False, C=None):
        if C is None:
            C = self.C
        nao = self.nao
        resp = self.resp_cphf if in_cphf else self.resp

        @timing
        def fx(X_):
            # For simplicity, shape of X should be (2, dim_prop, sk, sl)
            have_first_dim = len(X_[0].shape) >= 3
            prop_dim = int(np.prod(X_[0].shape[:-2]))
            restore_shape = list(X_[0].shape[:-2])
            X = [np.copy(X_[0]), np.copy(X_[1])]
            X[0] = X[0].reshape([prop_dim] + list(X_[0].shape[-2:]))
            X[1] = X[1].reshape([prop_dim] + list(X_[1].shape[-2:]))

            dm = np.zeros((2, prop_dim, nao, nao))
            dm[0] = C[0][:, sk[0]] @ X[0] @ C[0][:, sl[0]].T
            dm[1] = C[1][:, sk[1]] @ X[1] @ C[1][:, sl[1]].T
            dm += dm.swapaxes(-1, -2)
            if not have_first_dim:
                dm.shape = (2, nao, nao)

            ax_ao = resp(dm)
            ax_ao.shape = tuple([2] + restore_shape + [nao, nao])
            Ax = (
                C[0][:, si[0]].T @ ax_ao[0] @ C[0][:, sj[0]],
                C[1][:, si[1]].T @ ax_ao[1] @ C[1][:, sj[1]]
            )
            return Ax

        return fx

    @cached_property
    def U_1(self):
        B_1 = self.B_1
        S_1_mo = self.S_1_mo
        if not isinstance(S_1_mo, np.ndarray):
            S_1_mo = np.zeros_like(B_1)
        Ax0_Core = self.Ax0_Core
        sv, so = self.sv, self.so
        nocc, nvir, nmo = self.nocc, self.nvir, self.nmo
        e, eo, ev, mo_occ = self.e, self.eo, self.ev, self.mo_occ
        prop_dim = B_1.shape[1]
        # Calculate U_1_vo
        def fx(X):
            prop_dim = X.shape[0]
            X_alpha = X[:, :nocc[0] * nvir[0]].reshape((prop_dim, nvir[0], nocc[0]))
            X_beta = X[:, nocc[0] * nvir[0]:].reshape((prop_dim, nvir[1], nocc[1]))
            Ax = Ax0_Core(sv, so, sv, so, in_cphf=True)((X_alpha, X_beta))
            result = np.concatenate([Ax[0].reshape(prop_dim, -1), Ax[1].reshape(prop_dim, -1)], axis=1)
            return result
        U_1_vo = ucphf.solve(fx, e, mo_occ, (B_1[0, :, sv[0], so[0]], B_1[1, :, sv[1], so[1]]), max_cycle=100, tol=self.cphf_tol)[0]

        # Additional Iteration by newton_krylov
        def get_conv(U_1_vo):
            Ax_U = Ax0_Core(sv, so, sv, so, in_cphf=True)(U_1_vo)
            return (
                (ev[0][:, None] - eo[0][None, :]) * U_1_vo[0] + Ax_U[0] + B_1[0][:, sv[0], so[0]],
                (ev[1][:, None] - eo[1][None, :]) * U_1_vo[1] + Ax_U[1] + B_1[1][:, sv[1], so[1]],
            )

        # def vind(guess):
        #     U_1_vo_inner = (
        #         guess[:3 * nocc[0] * nvir[0]].reshape(3, nvir[0], nocc[0]),
        #         guess[3 * nocc[0] * nvir[0]:].reshape(3, nvir[1], nocc[1]))
        #     conv = get_conv(U_1_vo_inner)
        #     return np.concatenate([conv[0].ravel(), conv[1].ravel()])
        #
        # guess = (
        #     B_1[0, :, sv[0], so[0]] / (ev[0][:, None] - eo[0][None, :]),
        #     B_1[1, :, sv[1], so[1]] / (ev[1][:, None] - eo[1][None, :]))
        # guess = np.concatenate([guess[0].ravel(), guess[1].ravel()])
        # res = newton_krylov(vind, guess, f_tol=1e-10)
        # U_1_vo = (
        #     res[:3 * nocc[0] * nvir[0]].reshape(3, nvir[0], nocc[0]),
        #     res[3 * nocc[0] * nvir[0]:].reshape(3, nvir[1], nocc[1]))
        # Check sanity
        conv = get_conv(U_1_vo)
        if np.abs(conv[0]).max() > 1e-8 or np.abs(conv[1]).max() > 1e-8:
            msg = "\nget_U_1: CP-HF not converged well!\nMaximum deviation: " + str(np.abs(conv[0]).max()) + ", " + str(np.abs(conv[1]).max())
            warnings.warn(msg)
        # Build rest of U_1
        U_1 = np.zeros((2, prop_dim, nmo, nmo))
        if self.rotation:
            U_1 = - 0.5 * S_1_mo
            U_1[0, :, sv[0], so[0]] = U_1_vo[0]
            U_1[1, :, sv[1], so[1]] = U_1_vo[1]
            U_1[0, :, so[0], sv[0]] = - S_1_mo[0, :, so[0], sv[0]] - U_1_vo[0].swapaxes(-1, -2)
            U_1[1, :, so[1], sv[1]] = - S_1_mo[1, :, so[1], sv[1]] - U_1_vo[1].swapaxes(-1, -2)
        else:
            U_1[0, :, sv[0], so[0]] = U_1_vo[0]
            U_1[1, :, sv[1], so[1]] = U_1_vo[1]
            U_1[0, :, so[0], sv[0]] = - S_1_mo[0, :, so[0], sv[0]] - U_1_vo[0].swapaxes(-1, -2)
            U_1[1, :, so[1], sv[1]] = - S_1_mo[1, :, so[1], sv[1]] - U_1_vo[1].swapaxes(-1, -2)
            Ax_oo = Ax0_Core(so, so, sv, so)(U_1_vo)
            Ax_vv = Ax0_Core(sv, sv, sv, so)(U_1_vo)
            U_1[0, :, so[0], so[0]] = - (Ax_oo[0] + B_1[0, :, so[0], so[0]]) / (eo[0][:, None] - eo[0][None, :])
            U_1[1, :, so[1], so[1]] = - (Ax_oo[1] + B_1[1, :, so[1], so[1]]) / (eo[1][:, None] - eo[1][None, :])
            U_1[0, :, sv[0], sv[0]] = - (Ax_vv[0] + B_1[0, :, sv[0], sv[0]]) / (ev[0][:, None] - ev[0][None, :])
            U_1[1, :, sv[1], sv[1]] = - (Ax_vv[1] + B_1[1, :, sv[1], sv[1]]) / (ev[1][:, None] - ev[1][None, :])
            for p in range(nmo):
                U_1[:, :, p, p] = - S_1_mo[:, :, p, p] / 2
            U_1 -= (U_1 + U_1.swapaxes(-1, -2) + S_1_mo) / 2
            U_1 -= (U_1 + U_1.swapaxes(-1, -2) + S_1_mo) / 2

        return U_1

    @cached_property
    def resp(self) -> Callable:
        return _gen_uhf_response(self.scf_eng, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)

    @cached_property
    def resp_cphf(self) -> Callable:
        if self.xc_type == "HF":
            return self.resp
        else:
            mf = dft.RKS(self.mol)
            mf.xc = self.scf_eng.xc
            mf.grids = self.cphf_grids
            return _gen_uhf_response(mf, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)


class DerivOnceUNCDFT(DerivOnceUSCF, DerivOnceNCDFT):

    @cached_property
    def Z(self):
        so, sv = self.so, self.sv
        nocc, nvir = self.nocc, self.nvir
        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        F_0_mo = self.nc_deriv.F_0_mo
        def fx(X):
            X_ = (
                X[:, :nocc[0] * nvir[0]].reshape((nvir[0], nocc[0])),
                X[:, nocc[0] * nvir[0]:].reshape((nvir[1], nocc[1]))
            )
            return np.concatenate([v.ravel() for v in Ax0_Core(sv, so, sv, so, in_cphf=True)(X_)])
        Z = ucphf.solve(fx, e, mo_occ, (F_0_mo[0, None, sv[0], so[0]], F_0_mo[1, None, sv[1], so[1]]), max_cycle=100, tol=self.cphf_tol)[0]
        # output Z shape is (1, nvir, nocc), we remove the first dimension
        Z = (Z[0][0], Z[1][0])
        return Z


class DerivOnceUMP2(DerivOnceUSCF, DerivOnceMP2, ABC):

    @cached_property
    def D_iajb(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eo, ev = self.eo, self.ev
        D_iajb = (
            eo[0][:, None, None, None] - ev[0][None, :, None, None] + eo[0][None, None, :, None] - ev[0][None, None, None, :],
            eo[0][:, None, None, None] - ev[0][None, :, None, None] + eo[1][None, None, :, None] - ev[1][None, None, None, :],
            eo[1][:, None, None, None] - ev[1][None, :, None, None] + eo[1][None, None, :, None] - ev[1][None, None, None, :]
        )
        return D_iajb

    @cached_property
    def t_iajb(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        so, sv = self.so, self.sv
        eri0_mo = self.eri0_mo
        D_iajb = self.D_iajb
        t_iajb = (
            eri0_mo[0][so[0], sv[0], so[0], sv[0]] / D_iajb[0],
            eri0_mo[1][so[0], sv[0], so[1], sv[1]] / D_iajb[1],
            eri0_mo[2][so[1], sv[1], so[1], sv[1]] / D_iajb[2]
        )
        return t_iajb

    @cached_property
    def T_iajb(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_iajb = self.t_iajb
        cc, ss, os = self.cc, self.ss, self.os
        T_iajb = (
            0.5 * cc * ss * (t_iajb[0] - t_iajb[0].swapaxes(-1, -3)),
            cc * os * t_iajb[1],
            0.5 * cc * ss * (t_iajb[2] - t_iajb[2].swapaxes(-1, -3))
        )
        return T_iajb

    @cached_property
    def eng(self):
        T_iajb, t_iajb, D_iajb = self.T_iajb, self.t_iajb, self.D_iajb
        return self.scf_eng.e_tot + np.array([(T_iajb[i] * t_iajb[i] * D_iajb[i]).sum() for i in range(3)]).sum()

    @cached_property
    def W_I(self) -> np.ndarray:
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

    @cached_property
    def D_r_oovv(self) -> np.ndarray:
        nmo = self.nmo
        so, sv = self.so, self.sv
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        D_r_oovv = np.zeros((2, nmo, nmo))
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
        return D_r_oovv

    def _get_L(self) -> Tuple[np.ndarray, np.ndarray]:
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        eri0_mo = self.eri0_mo
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        L = Ax0_Core(sv, so, sa, sa)(self.D_r_oovv)
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

    @cached_property
    def D_r(self) -> np.ndarray:
        L = self.L
        D_r = np.copy(self.D_r_oovv)
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

    @cached_property
    def pdA_eri0_mo(self) -> np.ndarray:
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

    @cached_property
    def pdA_t_iajb(self) -> np.ndarray:
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
        return pdA_t_iajb

    @cached_property
    def pdA_T_iajb(self):
        pdA_t_iajb = self.pdA_t_iajb
        cc, ss, os = self.cc, self.ss, self.os
        pdA_T_iajb = (
            0.5 * cc * ss * (pdA_t_iajb[0] - pdA_t_iajb[0].swapaxes(-1, -3)),
            cc * os * pdA_t_iajb[1],
            0.5 * cc * ss * (pdA_t_iajb[2] - pdA_t_iajb[2].swapaxes(-1, -3))
        )
        return pdA_T_iajb

    @cached_property
    def pdA_D_r_oovv(self):
        so, sv = self.so, self.sv
        nmo = self.nmo
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        pdA_t_iajb, pdA_T_iajb = self.pdA_t_iajb, self.pdA_T_iajb

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

    @cached_property
    def pdA_W_I(self):
        so, sv = self.so, self.sv
        natm, nmo = self.natm, self.nmo
        pdA_T_iajb, T_iajb = self.pdA_T_iajb, self.T_iajb
        eri0_mo, pdA_eri0_mo = self.eri0_mo, self.pdA_eri0_mo
        pdA_W_I = np.zeros((2, pdA_T_iajb[0].shape[0], nmo, nmo))
        pdA_W_I[0, :, so[0], so[0]] = (
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[0], eri0_mo[0][so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[0], pdA_eri0_mo[0][:, so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("iakb, Ajakb -> Aij", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        pdA_W_I[1, :, so[1], so[1]] = (
                - 2 * np.einsum("Aiakb, jakb -> Aij", pdA_T_iajb[2], eri0_mo[2][so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("Akbia, kbja -> Aij", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iakb, Ajakb -> Aij", T_iajb[2], pdA_eri0_mo[2][:, so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("kbia, Akbja -> Aij", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        # vir-vir part
        pdA_W_I[0, :, sv[0], sv[0]] = (
                - 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[0], eri0_mo[0][so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[0], pdA_eri0_mo[0][:, so[0], sv[0], so[0], sv[0]])
                - 1 * np.einsum("iajc, Aibjc -> Aab", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        pdA_W_I[1, :, sv[1], sv[1]] = (
                - 2 * np.einsum("Aiajc, ibjc -> Aab", pdA_T_iajb[2], eri0_mo[2][so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("Ajcia, jcib -> Aab", pdA_T_iajb[1], eri0_mo[1][so[0], sv[0], so[1], sv[1]])
                - 2 * np.einsum("iajc, Aibjc -> Aab", T_iajb[2], pdA_eri0_mo[2][:, so[1], sv[1], so[1], sv[1]])
                - 1 * np.einsum("jcia, Ajcib -> Aab", T_iajb[1], pdA_eri0_mo[1][:, so[0], sv[0], so[1], sv[1]]))
        # vir-occ part
        pdA_W_I[0, :, sv[0], so[0]] = (
                - 4 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[0], eri0_mo[0][so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[1], eri0_mo[1][so[0], so[0], sv[1], so[1]])
                - 4 * np.einsum("jakb, Aijbk -> Aai", T_iajb[0], pdA_eri0_mo[0][:, so[0], so[0], sv[0], so[0]])
                - 2 * np.einsum("jakb, Aijbk -> Aai", T_iajb[1], pdA_eri0_mo[1][:, so[0], so[0], sv[1], so[1]]))
        pdA_W_I[1, :, sv[1], so[1]] = (
                - 4 * np.einsum("Ajakb, ijbk -> Aai", pdA_T_iajb[2], eri0_mo[2][so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("Akbja, bkij -> Aai", pdA_T_iajb[1], eri0_mo[1][sv[0], so[0], so[1], so[1]])
                - 4 * np.einsum("jakb, Aijbk -> Aai", T_iajb[2], pdA_eri0_mo[2][:, so[1], so[1], sv[1], so[1]])
                - 2 * np.einsum("kbja, Abkij -> Aai", T_iajb[1], pdA_eri0_mo[1][:, sv[0], so[0], so[1], so[1]]))
        return pdA_W_I


class DerivOnceUXDH(DerivOnceUMP2, DerivOnceUNCDFT, ABC):

    def _get_L(self):
        sv, so = self.sv, self.so
        L = super(DerivOnceUXDH, self)._get_L()
        L[0][:] += 2 * self.nc_deriv.F_0_mo[0][sv[0], so[0]]
        L[1][:] += 2 * self.nc_deriv.F_0_mo[1][sv[1], so[1]]
        return L

    @cached_property
    def eng(self):
        T_iajb, t_iajb, D_iajb = self.T_iajb, self.t_iajb, self.D_iajb
        eng = self.nc_deriv.scf_eng.energy_tot(dm=self.D) + np.array([(T_iajb[i] * t_iajb[i] * D_iajb[i]).sum() for i in range(3)]).sum()
        return eng

