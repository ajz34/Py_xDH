# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyscf utilities
from pyscf import grad
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceUSCF, GradSCF, DerivOnceUNCDFT, DerivOnceUMP2, DerivOnceUXDH
from pyxdh.Utilities import GridIterator, KernelHelper, timing, cached_property


class GradUSCF(DerivOnceUSCF, GradSCF):

    @cached_property
    def F_1_ao(self) -> np.ndarray:
        return np.array(self.scf_hess.make_h1(self.C, self.mo_occ)).reshape((2, self.natm * 3, self.nao, self.nao))

    def _get_E_1(self):
        mol = self.mol
        natm = self.natm
        cx, xc = self.cx, self.xc
        H_1_ao = self.H_1_ao
        eri1_ao = self.eri1_ao
        S_1_mo = self.S_1_mo
        F_0_mo = self.F_0_mo
        occ = self.occ
        D = self.D
        grids = self.grids

        E_1 = (
                + einsum("Auv, xuv -> A", H_1_ao, D)
                + 0.5 * einsum("Auvkl, yuv, xkl -> A", eri1_ao, D, D)
                - 0.5 * cx * einsum("Aukvl, xuv, xkl -> A", eri1_ao, D, D)
                - einsum("xApq, xpq, xp, xq -> A", S_1_mo, F_0_mo, occ, occ)
                + grad.rhf.grad_nuc(mol).reshape(-1)
        )

        # GGA part contiribution
        if self.xc_type == "GGA":
            grdit = zip(GridIterator(mol, grids, D[0], deriv=2), GridIterator(mol, grids, D[1], deriv=2))
            for grdh in grdit:
                kerh = KernelHelper(grdh, xc)
                E_1 += (
                    + einsum("g, Atg -> At", kerh.fr[0], grdh[0].A_rho_1)
                    + einsum("g, Atg -> At", kerh.fr[1], grdh[1].A_rho_1)
                    + 2 * einsum("g, rg, Atrg -> At", kerh.fg[0], grdh[0].rho_1, grdh[0].A_rho_2)
                    + 2 * einsum("g, rg, Atrg -> At", kerh.fg[2], grdh[1].rho_1, grdh[1].A_rho_2)
                    + 1 * einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[1].rho_1, grdh[0].A_rho_2)
                    + 1 * einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[0].rho_1, grdh[1].A_rho_2)
                ).reshape(-1)

        return E_1.reshape((natm, 3))

    def Ax1_Core(self, si, sj, sk, sl, reshape=True):

        C, Co = self.C, self.Co
        natm, nao = self.natm, self.nao
        cx = self.cx
        eri1_ao = self.eri1_ao

        @timing
        def fx(X_):
            if self.xc_type != "HF":
                raise NotImplementedError("DFT is not implemented!")

            if not isinstance(X_[0], np.ndarray):
                return 0

            have_first_dim = len(X_[0].shape) >= 3
            prop_dim = X_[0].shape[0] if have_first_dim else 1
            restore_shape = list(X_[0].shape[:-2])

            dmX = np.zeros((2, prop_dim, nao, nao))
            dmX[0] = C[0][:, sk[0]] @ X_[0] @ C[0][:, sl[0]].T
            dmX[1] = C[1][:, sk[1]] @ X_[1] @ C[1][:, sl[1]].T
            dmX += dmX.swapaxes(-1, -2)

            ax_ao = (
                + einsum("Auvkl, xBkl -> ABuv", eri1_ao, dmX)
                - cx * einsum("Aukvl, xBkl -> xABuv", eri1_ao, dmX)
            )
            ax_ao = (
                einsum("ABuv, ui, vj -> ABij", ax_ao[0], C[0][:, si[0]], C[0][:, sj[0]]),
                einsum("ABuv, ui, vj -> ABij", ax_ao[1], C[1][:, si[1]], C[1][:, sj[1]]),
            )
            ax_ao[0].shape = tuple([ax_ao[0].shape[0]] + restore_shape + list(ax_ao[0].shape[-2:]))
            ax_ao[1].shape = tuple([ax_ao[1].shape[0]] + restore_shape + list(ax_ao[1].shape[-2:]))
            return ax_ao
        return fx


class GradUNCDFT(DerivOnceUNCDFT, GradUSCF):

    @property
    def DerivOnceMethod(self):
        return GradUSCF

    def _get_E_1(self):
        natm = self.natm
        so, sv = self.so, self.sv
        B_1 = self.B_1
        Z = self.Z
        E_1 = (
            + self.nc_deriv.E_1
            + 2 * einsum("ai, Aai -> A", Z[0], B_1[0, :, sv[0], so[0]]).reshape((natm, 3))
            + 2 * einsum("ai, Aai -> A", Z[1], B_1[1, :, sv[1], so[1]]).reshape((natm, 3))
        )
        return E_1


class GradUMP2(DerivOnceUMP2, GradUSCF):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        D_r, B_1, W_I, S_1_mo, T_iajb, eri1_mo = self.D_r, self.B_1, self.W_I, self.S_1_mo, self.T_iajb, self.eri1_mo
        E_1 = (
            + np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
            + np.einsum("xpq, xApq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[0], eri1_mo[0][:, so[0], sv[0], so[0], sv[0]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[1], eri1_mo[1][:, so[0], sv[0], so[1], sv[1]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[2], eri1_mo[2][:, so[1], sv[1], so[1], sv[1]])
        ).reshape((natm, 3))
        E_1 += GradUSCF._get_E_1(self)
        return E_1


class GradUXDH(DerivOnceUXDH, GradUMP2, GradUNCDFT):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        D_r, B_1, W_I, S_1_mo, T_iajb, eri1_mo = self.D_r, self.B_1, self.W_I, self.S_1_mo, self.T_iajb, self.eri1_mo
        E_1 = (
            + np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
            + np.einsum("xpq, xApq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[0], eri1_mo[0][:, so[0], sv[0], so[0], sv[0]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[1], eri1_mo[1][:, so[0], sv[0], so[1], sv[1]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[2], eri1_mo[2][:, so[1], sv[1], so[1], sv[1]])
        ).reshape((natm, 3))
        E_1 += self.nc_deriv.E_1
        return E_1
