# python utilities
import numpy as np
from opt_einsum import contract as einsum
from warnings import warn
from functools import partial
# pyscf utilities
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceDFSCF, DerivOnceDFMP2, GradSCF, GradMP2
from pyxdh.Utilities import cached_property, GridIterator, KernelHelper, timing

# additional
einsum = partial(einsum, optimize="greedy")


class GradDFSCF(DerivOnceDFSCF, GradSCF):

    # region Auxiliary Basis Integral Generation

    @staticmethod
    def _gen_int2c2e_1(aux):
        int2c2e_ip1 = aux.intor("int2c2e_ip1")
        int2c2e_1 = np.zeros((aux.natm, 3, aux.nao, aux.nao))
        for A, (_, _, A0aux, A1aux) in enumerate(aux.aoslice_by_atom()):
            sAaux = slice(A0aux, A1aux)
            int2c2e_1[A, :, sAaux, :] -= int2c2e_ip1[:, sAaux, :]
            int2c2e_1[A, :, :, sAaux] -= int2c2e_ip1[:, sAaux, :].swapaxes(-1, -2)
        return int2c2e_1.reshape((aux.natm * 3, aux.nao, aux.nao))

    @staticmethod
    def _gen_int3c2e_1(mol, aux):
        int3c2e_ip1 = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")()
        int3c2e_ip2 = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")()
        int3c2e_1 = np.zeros((mol.natm, 3, mol.nao, mol.nao, aux.nao))
        for A in range(mol.natm):
            _, _, A0, A1 = mol.aoslice_by_atom()[A]
            _, _, A0aux, A1aux = aux.aoslice_by_atom()[A]
            sA, sAaux = slice(A0, A1), slice(A0aux, A1aux)
            int3c2e_1[A, :, sA, :, :] -= int3c2e_ip1[:, sA, :, :]
            int3c2e_1[A, :, :, sA, :] -= int3c2e_ip1[:, sA, :, :].swapaxes(-2, -3)
            int3c2e_1[A, :, :, :, sAaux] -= int3c2e_ip2[:, :, :, sAaux]
        return int3c2e_1.reshape((mol.natm * 3, mol.nao, mol.nao, aux.nao))

    # endregion Auxiliary Basis Integral Generation

    @cached_property
    def eri1_ao(self):
        warn("eri1 should not be called in density fitting module!", FutureWarning)
        return (
            + einsum("AμνP, κλP -> Aμνκλ", self.Y_ao_1_jk, self.Y_ao_jk)
            + einsum("μνP, AκλP -> Aμνκλ", self.Y_ao_jk, self.Y_ao_1_jk))

    def Ax1_Core_(self, si, sj, sk, sl, reshape=True):

        C, Co = self.C, self.Co
        natm, nao = self.natm, self.nao
        mol = self.mol
        cx = self.cx
        so = self.so

        dmU = C @ self.U_1[:, :, so] @ Co.T
        dmU += dmU.swapaxes(-1, -2)
        dmU.shape = (natm, 3, nao, nao)

        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        @timing
        def fx(X_):
            if not isinstance(X_, np.ndarray):
                return 0
            X = X_.copy()  # type: np.ndarray
            shape1 = list(X.shape)
            X.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dmX = X
                if dmX.shape[-2] != nao or dmX.shape[-1] != nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dmX = C[:, sk] @ X @ C[:, sl].T
            dmX += dmX.transpose((0, 2, 1))

            ax_ao = np.zeros((natm * 3, dmX.shape[0], nao, nao))

            # ( 4 (μν|κλ)^A - 2 cx (μκ|νλ)^A ) * D_{κλ}
            ax_ao += 4 * einsum("AμνP, κλP, Bκλ -> ABμν", self.Y_ao_1_jk, self.Y_ao_jk, dmX)
            ax_ao += 4 * einsum("μνP, AκλP, Bκλ -> ABμν", self.Y_ao_jk, self.Y_ao_1_jk, dmX)
            ax_ao -= 2 * cx * einsum("AμκP, νλP, Bκλ -> ABμν", self.Y_ao_1_jk, self.Y_ao_jk, dmX)
            ax_ao -= 2 * cx * einsum("μκP, AνλP, Bκλ -> ABμν", self.Y_ao_jk, self.Y_ao_1_jk, dmX)

            ax_ao.shape = (natm, 3, dmX.shape[0], nao, nao)

            # GGA Part
            if self.xc_type == "GGA":
                # TODO: change GGA part. this part of code is copied from grad_r.
                grdit = GridIterator(self.mol, self.grids, self.D, deriv=3, memory=self.grdit_memory)
                for grdh in grdit:
                    kerh = KernelHelper(grdh, self.xc, deriv=3)
                    # Define some kernel and density derivative alias
                    pd_frr = kerh.frrr * grdh.A_rho_1 + kerh.frrg * grdh.A_gamma_1
                    pd_frg = kerh.frrg * grdh.A_rho_1 + kerh.frgg * grdh.A_gamma_1
                    pd_fgg = kerh.frgg * grdh.A_rho_1 + kerh.fggg * grdh.A_gamma_1
                    pd_fg = kerh.frg * grdh.A_rho_1 + kerh.fgg * grdh.A_gamma_1
                    pd_rho_1 = grdh.A_rho_2

                    # Form dmX density grid
                    rho_X_0 = np.array([grdh.get_rho_0(dm) for dm in dmX])
                    rho_X_1 = np.array([grdh.get_rho_1(dm) for dm in dmX])
                    pd_rho_X_0 = np.array([grdh.get_A_rho_1(dm) for dm in dmX]).transpose((1, 2, 0, 3))
                    pd_rho_X_1 = np.array([grdh.get_A_rho_2(dm) for dm in dmX]).transpose((1, 2, 0, 3, 4))

                    # Define temporary intermediates
                    tmp_M_0 = (
                            + einsum("g, Bg -> Bg", kerh.frr, rho_X_0)
                            + 2 * einsum("g, wg, Bwg -> Bg", kerh.frg, grdh.rho_1, rho_X_1)
                    )
                    tmp_M_1 = (
                            + 4 * einsum("g, Bg, rg -> Brg", kerh.frg, rho_X_0, grdh.rho_1)
                            + 8 * einsum("g, wg, Bwg, rg -> Brg", kerh.fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 4 * einsum("g, Brg -> Brg", kerh.fg, rho_X_1)
                    )
                    pd_tmp_M_0 = (
                            + einsum("Atg, Bg -> AtBg", pd_frr, rho_X_0)
                            + einsum("g, AtBg -> AtBg", kerh.frr, pd_rho_X_0)
                            + 2 * einsum("Atg, wg, Bwg -> AtBg", pd_frg, grdh.rho_1, rho_X_1)
                            + 2 * einsum("g, Atwg, Bwg -> AtBg", kerh.frg, pd_rho_1, rho_X_1)
                            + 2 * einsum("g, wg, AtBwg -> AtBg", kerh.frg, grdh.rho_1, pd_rho_X_1)
                    )
                    pd_tmp_M_1 = (
                            + 4 * einsum("Atg, Bg, rg -> AtBrg", pd_frg, rho_X_0, grdh.rho_1)
                            + 4 * einsum("g, Bg, Atrg -> AtBrg", kerh.frg, rho_X_0, pd_rho_1)
                            + 4 * einsum("g, AtBg, rg -> AtBrg", kerh.frg, pd_rho_X_0, grdh.rho_1)
                            + 8 * einsum("Atg, wg, Bwg, rg -> AtBrg", pd_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 8 * einsum("g, Atwg, Bwg, rg -> AtBrg", kerh.fgg, pd_rho_1, rho_X_1, grdh.rho_1)
                            + 8 * einsum("g, wg, Bwg, Atrg -> AtBrg", kerh.fgg, grdh.rho_1, rho_X_1, pd_rho_1)
                            + 8 * einsum("g, wg, AtBwg, rg -> AtBrg", kerh.fgg, grdh.rho_1, pd_rho_X_1, grdh.rho_1)
                            + 4 * einsum("Atg, Brg -> AtBrg", pd_fg, rho_X_1)
                            + 4 * einsum("g, AtBrg -> AtBrg", kerh.fg, pd_rho_X_1)
                    )

                    contrib1 = np.zeros((natm, 3, dmX.shape[0], nao, nao))
                    contrib1 += einsum("AtBg, gu, gv -> AtBuv", pd_tmp_M_0, grdh.ao_0, grdh.ao_0)
                    contrib1 += einsum("AtBrg, rgu, gv -> AtBuv", pd_tmp_M_1, grdh.ao_1, grdh.ao_0)
                    contrib1 += contrib1.swapaxes(-1, -2)

                    tmp_contrib = (
                            - 2 * einsum("Bg, tgu, gv -> tBuv", tmp_M_0, grdh.ao_1, grdh.ao_0)
                            - einsum("Brg, trgu, gv -> tBuv", tmp_M_1, grdh.ao_2, grdh.ao_0)
                            - einsum("Brg, tgu, rgv -> tBuv", tmp_M_1, grdh.ao_1, grdh.ao_1)
                    )

                    contrib2 = np.zeros((natm, 3, dmX.shape[0], nao, nao))
                    for A in range(natm):
                        sA = self.mol_slice(A)
                        contrib2[A, :, :, sA] += tmp_contrib[:, :, sA]

                    contrib2 += contrib2.swapaxes(-1, -2)

                    # U contribution to \partial_{A_t} A
                    rho_U_0 = einsum("Atuv, gu, gv -> Atg", dmU, grdh.ao_0, grdh.ao_0)
                    rho_U_1 = 2 * einsum("Atuv, rgu, gv -> Atrg", dmU, grdh.ao_1, grdh.ao_0)
                    gamma_U_0 = 2 * einsum("rg, Atrg -> Atg", grdh.rho_1, rho_U_1)
                    pdU_frr = kerh.frrr * rho_U_0 + kerh.frrg * gamma_U_0
                    pdU_frg = kerh.frrg * rho_U_0 + kerh.frgg * gamma_U_0
                    pdU_fgg = kerh.frgg * rho_U_0 + kerh.fggg * gamma_U_0
                    pdU_fg = kerh.frg * rho_U_0 + kerh.fgg * gamma_U_0
                    pdU_rho_1 = rho_U_1
                    pdU_tmp_M_0 = (
                            + einsum("Atg, Bg -> AtBg", pdU_frr, rho_X_0)
                            + 2 * einsum("Atg, wg, Bwg -> AtBg", pdU_frg, grdh.rho_1, rho_X_1)
                            + 2 * einsum("g, Atwg, Bwg -> AtBg", kerh.frg, pdU_rho_1, rho_X_1)
                    )
                    pdU_tmp_M_1 = (
                            + 4 * einsum("Atg, Bg, rg -> AtBrg", pdU_frg, rho_X_0, grdh.rho_1)
                            + 4 * einsum("g, Bg, Atrg -> AtBrg", kerh.frg, rho_X_0, pdU_rho_1)
                            + 8 * einsum("Atg, wg, Bwg, rg -> AtBrg", pdU_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 8 * einsum("g, Atwg, Bwg, rg -> AtBrg", kerh.fgg, pdU_rho_1, rho_X_1, grdh.rho_1)
                            + 8 * einsum("g, wg, Bwg, Atrg -> AtBrg", kerh.fgg, grdh.rho_1, rho_X_1, pdU_rho_1)
                            + 4 * einsum("Atg, Brg -> AtBrg", pdU_fg, rho_X_1)
                    )

                    contrib3 = np.zeros((natm, 3, dmX.shape[0], nao, nao))
                    contrib3 += einsum("AtBg, gu, gv -> AtBuv", pdU_tmp_M_0, grdh.ao_0, grdh.ao_0)
                    contrib3 += einsum("AtBrg, rgu, gv -> AtBuv", pdU_tmp_M_1, grdh.ao_1, grdh.ao_0)
                    contrib3 += contrib3.swapaxes(-1, -2)

                    ax_ao += contrib1 + contrib2 + contrib3

            ax_ao.shape = (natm * 3, dmX.shape[0], nao, nao)

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

    def _get_E_1(self):
        so = self.so
        return (
            + einsum("Auv, uv -> A", self.H_1_ao, self.D)
            + 1.0 * einsum("AuvP, klP, uv, kl -> A", self.Y_ao_1_jk, self.Y_ao_jk, self.D, self.D)
            - 0.5 * einsum("AuvP, klP, uk, vl -> A", self.Y_ao_1_jk, self.Y_ao_jk, self.D, self.D)
            - 2 * einsum("ij, Aij -> A", self.F_0_mo[so, so], self.S_1_mo[:, so, so])
        ).reshape((self.mol.natm, 3)) + self.scf_grad.grad_nuc()


class GradDFMP2(DerivOnceDFMP2, GradDFSCF, GradMP2):

    def _get_E_1(self):
        natm = self.natm
        E_1 = (
            + einsum("pq, Apq -> A", self.D_r, self.B_1)
            + einsum("pq, Apq -> A", self.W_I, self.S_1_mo)
            + 2 * einsum("iajb, AiaP, jbP -> A", self.T_iajb, self.Y_ia_1_ri, self.Y_ia_ri)
            + 2 * einsum("iajb, iaP, AjbP -> A", self.T_iajb, self.Y_ia_ri, self.Y_ia_1_ri)
        ).reshape(natm, 3)
        E_1 += GradDFSCF._get_E_1(self)
        return E_1
