# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivOnce.deriv_once_r import DerivOnceSCF, DerivOnceNCDFT, DerivOnceMP2, DerivOnceXDH
from pyxdh.Utilities import GridIterator, KernelHelper, cached_property


class DipoleSCF(DerivOnceSCF):

    def __init__(self, config):
        super(DipoleSCF, self).__init__(config)
        self.components = (config.get("components", (0, 1, 2)), )

    def Ax1_Core(self, si, sj, sk, sl, reshape=True):

        C, Co = self.C, self.Co
        nao = self.nao
        so = self.so
        num_components = len(self.components[0])

        dmU = C @ self.U_1[:, :, so] @ Co.T
        dmU += dmU.swapaxes(-1, -2)
        dmU.shape = (num_components, nao, nao)

        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        def fx(X_):

            # Method is only used in DFT
            if self.xc_type == "HF":
                return 0

            if not isinstance(X_, np.ndarray):
                return 0
            X = X_.copy()  # type: np.ndarray
            shape1 = list(X.shape)
            X.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dm = X
                if dm.shape[-2] != nao or dm.shape[-1] != nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dm = C[:, sk] @ X @ C[:, sl].T
            dm += dm.transpose((0, 2, 1))

            ax_ao = np.zeros((num_components, dm.shape[0], nao, nao))

            # Actual calculation

            grdit = GridIterator(self.mol, self.grids, self.D, deriv=3, memory=self.grdit_memory)
            for grdh in grdit:
                kerh = KernelHelper(grdh, self.xc, deriv=3)

                # Form dmX density grid
                rho_X_0 = np.array([grdh.get_rho_0(dmX) for dmX in dm])
                rho_X_1 = np.array([grdh.get_rho_1(dmX) for dmX in dm])

                # U contribution to \partial_{A_t} A
                rho_U_0 = einsum("Auv, gu, gv -> Ag", dmU, grdh.ao_0, grdh.ao_0)
                rho_U_1 = 2 * einsum("Auv, rgu, gv -> Arg", dmU, grdh.ao_1, grdh.ao_0)
                gamma_U_0 = 2 * einsum("rg, Arg -> Ag", grdh.rho_1, rho_U_1)
                pdU_frr = kerh.frrr * rho_U_0 + kerh.frrg * gamma_U_0
                pdU_frg = kerh.frrg * rho_U_0 + kerh.frgg * gamma_U_0
                pdU_fgg = kerh.frgg * rho_U_0 + kerh.fggg * gamma_U_0
                pdU_fg = kerh.frg * rho_U_0 + kerh.fgg * gamma_U_0
                pdU_rho_1 = rho_U_1
                pdU_tmp_M_0 = (
                        + einsum("Ag, Bg -> ABg", pdU_frr, rho_X_0)
                        + 2 * einsum("Ag, wg, Bwg -> ABg", pdU_frg, grdh.rho_1, rho_X_1)
                        + 2 * einsum("g, Awg, Bwg -> ABg", kerh.frg, pdU_rho_1, rho_X_1)
                )
                pdU_tmp_M_1 = (
                        + 4 * einsum("Ag, Bg, rg -> ABrg", pdU_frg, rho_X_0, grdh.rho_1)
                        + 4 * einsum("g, Bg, Arg -> ABrg", kerh.frg, rho_X_0, pdU_rho_1)
                        + 8 * einsum("Ag, wg, Bwg, rg -> ABrg", pdU_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                        + 8 * einsum("g, Awg, Bwg, rg -> ABrg", kerh.fgg, pdU_rho_1, rho_X_1, grdh.rho_1)
                        + 8 * einsum("g, wg, Bwg, Arg -> ABrg", kerh.fgg, grdh.rho_1, rho_X_1, pdU_rho_1)
                        + 4 * einsum("Ag, Brg -> ABrg", pdU_fg, rho_X_1)
                )

                contrib3 = np.zeros((num_components, dm.shape[0], nao, nao))
                contrib3 += einsum("ABg, gu, gv -> ABuv", pdU_tmp_M_0, grdh.ao_0, grdh.ao_0)
                contrib3 += einsum("ABrg, rgu, gv -> ABuv", pdU_tmp_M_1, grdh.ao_1, grdh.ao_0)
                contrib3 += contrib3.swapaxes(-1, -2)

                ax_ao += contrib3

            if not sij_none:
                ax_ao = einsum("ABuv, ui, vj -> ABij", ax_ao, C[:, si], C[:, sj])
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.insert(0, ax_ao.shape[0])
                shape1.append(ax_ao.shape[-2])
                shape1.append(ax_ao.shape[-1])
                ax_ao.shape = shape1

            return ax_ao

        return fx

    @cached_property
    def H_1_ao(self):
        return - self.mol.intor("int1e_r")[self.components]

    @cached_property
    def F_1_ao(self):
        return self.H_1_ao

    @cached_property
    def S_1_ao(self):
        return 0

    @cached_property
    def eri1_ao(self):
        return 0

    def _get_E_1(self):
        mol = self.mol
        H_1_ao = self.H_1_ao
        D = self.D

        dip_elec = einsum("Apq, pq -> A", H_1_ao, D)
        dip_nuc = einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())

        return dip_elec + dip_nuc


class DipoleNCDFT(DerivOnceNCDFT, DipoleSCF):

    @property
    def DerivOnceMethod(self):
        return DipoleSCF

    def _get_E_1(self):
        so, sv = self.so, self.sv
        B_1 = self.B_1
        Z = self.Z
        E_1 = 4 * einsum("ai, Aai -> A", Z, B_1[:, sv, so])
        E_1 += self.nc_deriv.E_1
        return E_1


class DipoleMP2(DerivOnceMP2, DipoleSCF):

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += DipoleSCF._get_E_1(self)
        return E_1


class DipoleXDH(DerivOnceXDH, DipoleMP2, DipoleNCDFT):

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += self.nc_deriv.E_1
        return E_1
