import numpy as np
from abc import ABC
from functools import partial
import os
import warnings
from pyscf.scf._response_functions import _gen_uhf_response
from scipy.optimize import newton_krylov

from pyscf import dft, grad, hessian
from pyscf.scf import ucphf

from pyxdh.DerivOnce.deriv_once_r import DerivOnceSCF, DerivOnceNCDFT
from pyxdh.Utilities import GridIterator, KernelHelper, timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceUSCF(DerivOnceSCF, ABC):

    def initialization_scf(self):
        if self.init_scf:
            self.mo_occ = self.scf_eng.mo_occ
            self.C = self.scf_eng.mo_coeff
            self.e = self.scf_eng.mo_energy
            self.nocc = self.mol.nelec
        return

    def initialization_pyscf(self):
        if (self.scf_eng.mo_coeff is NotImplemented or self.scf_eng.mo_coeff is None) and self.init_scf:
            self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        if isinstance(self.scf_eng, dft.rks.RKS) or isinstance(self.scf_eng, dft.uks.UKS):
            self.xc = self.scf_eng.xc
            self.grids = self.scf_eng.grids
            self.xc_type = dft.libxc.xc_type(self.xc)
            self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
            self.scf_grad = grad.uks.Gradients(self.scf_eng)
            self.scf_hess = hessian.uks.Hessian(self.scf_eng)
        else:
            self.scf_grad = grad.UHF(self.scf_eng)
            self.scf_hess = hessian.UHF(self.scf_eng)
        return

    @property
    def nvir(self):
        return self.nmo - self.nocc[0], self.nmo - self.nocc[1]

    @property
    def so(self):
        return slice(0, self.nocc[0]), slice(0, self.nocc[1])

    @property
    def sv(self):
        return slice(self.nocc[0], self.nmo), slice(self.nocc[1], self.nmo)

    @property
    def sa(self):
        return slice(0, self.nmo), slice(0, self.nmo)

    @property
    def Co(self):
        return self.C[0, :, self.so[0]], self.C[1, :, self.so[1]]

    @property
    def Cv(self):
        return self.C[0, :, self.sv[0]], self.C[1, :, self.sv[1]]

    @property
    def eo(self):
        return self.e[0, self.so[0]], self.e[1, self.so[1]]

    @property
    def ev(self):
        return self.e[0, self.sv[0]], self.e[1, self.sv[1]]

    @property
    def U_1_vo(self):
        return self.U_1[0, :, self.sv[0], self.so[0]], self.U_1[1, :, self.sv[1], self.so[1]]

    @property
    def U_1_ov(self):
        return self.U_1[0, :, self.so[0], self.sv[0]], self.U_1[1, :, self.so[1], self.sv[1]]

    def _get_D(self):
        return np.einsum("xup, xp, xvp -> xuv", self.C, self.occ, self.C)

    def _get_H_0_mo(self):
        return np.einsum("xup, uv, xvq -> xpq", self.C, self.H_0_ao, self.C)

    def _get_S_0_mo(self):
        return np.einsum("xup, uv, xvq -> xpq", self.C, self.S_0_ao, self.C)

    def _get_eri0_mo(self):
        nmo = self.nmo
        C = self.C
        eri0_ao = self.eri0_ao
        eri0_mo = np.zeros((3, nmo, nmo, nmo, nmo))
        eri0_mo[0] = np.einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[0], C[0], C[0], C[0])
        eri0_mo[1] = np.einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[0], C[0], C[1], C[1])
        eri0_mo[2] = np.einsum("uvkl, up, vq, kr, ls -> pqrs", eri0_ao, C[1], C[1], C[1], C[1])
        return eri0_mo

    def _get_F_0_mo(self):
        return np.einsum("xup, xuv, xvq -> xpq", self.C, self.F_0_ao, self.C)

    def _get_H_1_mo(self):
        if not isinstance(self.H_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, xup, xvq -> xApq", self.H_1_ao, self.C, self.C)

    def _get_S_1_mo(self):
        if not isinstance(self.S_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, xup, xvq -> xApq", self.S_1_ao, self.C, self.C)

    def _get_eri1_mo(self):
        nmo = self.nmo
        C = self.C
        eri1_ao = self.eri1_ao
        eri1_mo = np.zeros((3, eri1_ao.shape[0], nmo, nmo, nmo, nmo))
        eri1_mo[0] = np.einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[0], C[0], C[0], C[0])
        eri1_mo[1] = np.einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[0], C[0], C[1], C[1])
        eri1_mo[2] = np.einsum("Auvkl, up, vq, kr, ls -> Apqrs", eri1_ao, C[1], C[1], C[1], C[1])
        return eri1_mo

    def _get_F_1_mo(self):
        if not isinstance(self.F_1_ao, np.ndarray):
            return 0
        return np.einsum("xAuv, xup, xvq -> xApq", self.F_1_ao, self.C, self.C)

    @property
    def resp(self):
        if self._resp is NotImplemented:
            self._resp = _gen_uhf_response(self.scf_eng, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)
        return self._resp

    def _get_B_1(self):
        sa = self.sa
        so = self.so

        B_1 = self.F_1_mo.copy()
        if isinstance(self.S_1_mo, np.ndarray):
            B_1 += (
                - np.einsum("xApq, xq -> xApq", self.S_1_mo, self.e)
                - 0.5 * np.array(self.Ax0_Core(sa, sa, so, so)((self.S_1_mo[0, :, so[0], so[0]], self.S_1_mo[1, :, so[1], so[1]])))
            )
        return B_1

    def _get_pdA_F_0_mo(self):
        F_1_mo = self.F_1_mo
        U_1 = self.U_1
        e = self.e
        Ax0_Core = self.Ax0_Core
        so, sa = self.so, self.sa

        pdA_F_0_mo = (
            + F_1_mo
            + np.einsum("xApq, xp -> xApq", U_1, e)
            + np.einsum("xAqp, xq -> xApq", U_1, e)
            + Ax0_Core(sa, sa, sa, so)((U_1[0, :, :, so[0]], U_1[1, :, :, so[1]]))
        )
        return pdA_F_0_mo

    @property
    def resp_cphf(self):
        if self._resp_cphf is NotImplemented:
            if self.xc_type == "HF":
                self._resp_cphf = self.resp
            else:
                mf = dft.RKS(self.mol)
                mf.xc = self.scf_eng.xc
                mf.grids = self.cphf_grids
                self._resp_cphf = _gen_uhf_response(mf, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)
        return self._resp_cphf

    def Ax0_Core(self, si, sj, sk, sl, reshape=True, in_cphf=False, C=None):
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

    def _get_U_1(self):
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
        # def fx(X):
        #     prop_dim = X.shape[0]
        #     X_alpha = X[:, :nocc[0] * nvir[0]].reshape((prop_dim, nvir[0], nocc[0]))
        #     X_beta = X[:, nocc[0] * nvir[0]:].reshape((prop_dim, nvir[1], nocc[1]))
        #     Ax = Ax0_Core(sv, so, sv, so, in_cphf=True)((X_alpha, X_beta))
        #     result = np.concatenate([Ax[0].reshape(prop_dim, -1), Ax[1].reshape(prop_dim, -1)], axis=1)
        #     return result
        # U_1_vo = ucphf.solve(fx, e, mo_occ, (B_1[0, :, sv[0], so[0]], B_1[1, :, sv[1], so[1]]), max_cycle=100, tol=self.cphf_tol)[0]
        # Additional Iteration by newton_krylov
        def get_conv(U_1_vo):
            Ax_U = Ax0_Core(sv, so, sv, so, in_cphf=True)(U_1_vo)
            conv = (
                (ev[0][:, None] - eo[0][None, :]) * U_1_vo[0] + Ax_U[0] + B_1[0][:, sv[0], so[0]],
                (ev[1][:, None] - eo[1][None, :]) * U_1_vo[1] + Ax_U[1] + B_1[1][:, sv[1], so[1]],
            )
            return conv

        def vind(guess):
            U_1_vo_inner = (
                guess[:3 * nocc[0] * nvir[0]].reshape(3, nvir[0], nocc[0]),
                guess[3 * nocc[0] * nvir[0]:].reshape(3, nvir[1], nocc[1]))
            conv = get_conv(U_1_vo_inner)
            return np.concatenate([conv[0].ravel(), conv[1].ravel()])

        guess = (
            B_1[0, :, sv[0], so[0]] / (ev[0][:, None] - eo[0][None, :]),
            B_1[1, :, sv[1], so[1]] / (ev[1][:, None] - eo[1][None, :]))
        guess = np.concatenate([guess[0].ravel(), guess[1].ravel()])
        res = newton_krylov(vind, guess, f_tol=1e-10)
        U_1_vo = (
            res[:3 * nocc[0] * nvir[0]].reshape(3, nvir[0], nocc[0]),
            res[3 * nocc[0] * nvir[0]:].reshape(3, nvir[1], nocc[1]))
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


class DerivOnceUNCDFT(DerivOnceUSCF, DerivOnceNCDFT):

    def _get_Z(self):
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

