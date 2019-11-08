import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce.deriv_once_scf import DerivOnceSCF, DerivOnceNCDFT
from pyxdh.Utilities import GridIterator, KernelHelper


MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


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
                rho_U_0 = np.einsum("Auv, gu, gv -> Ag", dmU, grdh.ao_0, grdh.ao_0)
                rho_U_1 = 2 * np.einsum("Auv, rgu, gv -> Arg", dmU, grdh.ao_1, grdh.ao_0)
                gamma_U_0 = 2 * np.einsum("rg, Arg -> Ag", grdh.rho_1, rho_U_1)
                pdU_frr = kerh.frrr * rho_U_0 + kerh.frrg * gamma_U_0
                pdU_frg = kerh.frrg * rho_U_0 + kerh.frgg * gamma_U_0
                pdU_fgg = kerh.frgg * rho_U_0 + kerh.fggg * gamma_U_0
                pdU_fg = kerh.frg * rho_U_0 + kerh.fgg * gamma_U_0
                pdU_rho_1 = rho_U_1
                pdU_tmp_M_0 = (
                        + np.einsum("Ag, Bg -> ABg", pdU_frr, rho_X_0)
                        + 2 * np.einsum("Ag, wg, Bwg -> ABg", pdU_frg, grdh.rho_1, rho_X_1)
                        + 2 * np.einsum("g, Awg, Bwg -> ABg", kerh.frg, pdU_rho_1, rho_X_1)
                )
                pdU_tmp_M_1 = (
                        + 4 * np.einsum("Ag, Bg, rg -> ABrg", pdU_frg, rho_X_0, grdh.rho_1)
                        + 4 * np.einsum("g, Bg, Arg -> ABrg", kerh.frg, rho_X_0, pdU_rho_1)
                        + 8 * np.einsum("Ag, wg, Bwg, rg -> ABrg", pdU_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                        + 8 * np.einsum("g, Awg, Bwg, rg -> ABrg", kerh.fgg, pdU_rho_1, rho_X_1, grdh.rho_1)
                        + 8 * np.einsum("g, wg, Bwg, Arg -> ABrg", kerh.fgg, grdh.rho_1, rho_X_1, pdU_rho_1)
                        + 4 * np.einsum("Ag, Brg -> ABrg", pdU_fg, rho_X_1)
                )

                contrib3 = np.zeros((num_components, dm.shape[0], nao, nao))
                contrib3 += np.einsum("ABg, gu, gv -> ABuv", pdU_tmp_M_0, grdh.ao_0, grdh.ao_0)
                contrib3 += np.einsum("ABrg, rgu, gv -> ABuv", pdU_tmp_M_1, grdh.ao_1, grdh.ao_0)
                contrib3 += contrib3.swapaxes(-1, -2)

                ax_ao += contrib3

            if not sij_none:
                ax_ao = np.einsum("ABuv, ui, vj -> ABij", ax_ao, C[:, si], C[:, sj])
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.insert(0, ax_ao.shape[0])
                shape1.append(ax_ao.shape[-2])
                shape1.append(ax_ao.shape[-1])
                ax_ao.shape = shape1

            return ax_ao

        return fx

    def _get_H_1_ao(self):
        return - self.mol.intor("int1e_r")[self.components]

    def _get_F_1_ao(self):
        return self.H_1_ao

    def _get_S_1_ao(self):
        return 0

    def _get_eri1_ao(self):
        return 0

    def _get_E_1(self):
        mol = self.mol
        H_1_ao = self.H_1_ao
        D = self.D

        dip_elec = np.einsum("Apq, pq -> A", H_1_ao, D)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())

        return dip_elec + dip_nuc


class DipoleNCDFT(DerivOnceNCDFT, DipoleSCF):

    @property
    def DerivOnceMethod(self):
        return DipoleSCF

    def _get_E_1(self):
        so, sv = self.so, self.sv
        B_1 = self.B_1
        Z = self.Z
        E_1 = 4 * np.einsum("ai, Aai -> A", Z, B_1[:, sv, so])
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_DipoleSCF:

    @staticmethod
    def valid_assert(config, resource_path):
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface
        helper = DipoleSCF(config)
        assert (np.allclose(helper.E_1, helper.scf_eng.dip_moment(unit="A.U."), atol=1e-6, rtol=1e-4))
        formchk = FormchkInterface(resource_filename("pyxdh", resource_path))
        assert (np.allclose(helper.E_1, formchk.dipole(), atol=1e-6, rtol=1e-4))

    def test_SCF_dipole(self):

        from pyxdh.Utilities.test_molecules import Mol_H2O2

        H2O2 = Mol_H2O2()
        grids_cphf = H2O2.gen_grids(50, 194)
        self.valid_assert({"scf_eng": H2O2.hf_eng}, "Validation/gaussian/H2O2-HF-freq.fchk")
        self.valid_assert({"scf_eng": H2O2.gga_eng, "cphf_grids": grids_cphf}, "Validation/gaussian/H2O2-B3LYP-freq.fchk")

    def test_HF_B3LYP_dipole(self):

        import pickle
        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng, "nc_eng": H2O2.gga_eng}
        helper = DipoleNCDFT(config)

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/ncdft_derivonce_hf_b3lyp.dat"), "rb") as f:
            ref_dipole = pickle.load(f)["dipole"]
        assert (np.allclose(helper.E_1, ref_dipole, atol=1e-6, rtol=1e-4))
