import numpy as np
from functools import partial
import os

from pyscf import grad

from pyxdh.DerivOnce import DerivOnceUSCF, GradSCF, DerivOnceUNCDFT
from pyxdh.Utilities import GridIterator, KernelHelper, timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradUSCF(DerivOnceUSCF, GradSCF):

    def _get_F_1_ao(self):
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
        grdit_memory = self.grdit_memory

        E_1 = (
                + np.einsum("Auv, xuv -> A", H_1_ao, D)
                + 0.5 * np.einsum("Auvkl, yuv, xkl -> A", eri1_ao, D, D)
                - 0.5 * cx * np.einsum("Aukvl, xuv, xkl -> A", eri1_ao, D, D)
                - np.einsum("xApq, xpq, xp, xq -> A", S_1_mo, F_0_mo, occ, occ)
                + grad.rhf.grad_nuc(mol).reshape(-1)
        )

        # GGA part contiribution
        if self.xc_type == "GGA":
            grdit = zip(GridIterator(mol, grids, D[0], deriv=2), GridIterator(mol, grids, D[1], deriv=2))
            for grdh in grdit:
                kerh = KernelHelper(grdh, xc)
                E_1 += (
                    + np.einsum("g, Atg -> At", kerh.fr[0], grdh[0].A_rho_1)
                    + np.einsum("g, Atg -> At", kerh.fr[1], grdh[1].A_rho_1)
                    + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg[0], grdh[0].rho_1, grdh[0].A_rho_2)
                    + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg[2], grdh[1].rho_1, grdh[1].A_rho_2)
                    + 1 * np.einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[1].rho_1, grdh[0].A_rho_2)
                    + 1 * np.einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[0].rho_1, grdh[1].A_rho_2)
            ).reshape(-1)

        return E_1.reshape((natm, 3))

    def Ax1_Core(self, si, sj, sk, sl, reshape=True):

        C, Co = self.C, self.Co
        natm, nao = self.natm, self.nao
        mol = self.mol
        cx = self.cx
        so = self.so
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
                + np.einsum("Auvkl, xBkl -> ABuv", eri1_ao, dmX)
                - cx * np.einsum("Aukvl, xBkl -> xABuv", eri1_ao, dmX)
            )
            ax_ao = (
                np.einsum("ABuv, ui, vj -> ABij", ax_ao[0], C[0][:, si[0]], C[0][:, sj[0]]),
                np.einsum("ABuv, ui, vj -> ABij", ax_ao[1], C[1][:, si[1]], C[1][:, sj[1]]),
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
            + 2 * np.einsum("ai, Aai -> A", Z[0], B_1[0, :, sv[0], so[0]]).reshape((natm, 3))
            + 2 * np.einsum("ai, Aai -> A", Z[1], B_1[1, :, sv[1], so[1]]).reshape((natm, 3))
        )
        return E_1


class Test_GradUSCF:

    def test_UHF_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = GradUSCF({"scf_eng": CH3.hf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))

    def test_UB3LYP_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = GradUSCF({"scf_eng": CH3.gga_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-B3LYP-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-6, rtol=1e-4))

    def test_UHF_UB3LYP_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        import pickle

        CH3 = Mol_CH3()
        helper = GradUNCDFT({"scf_eng": CH3.hf_eng, "nc_eng": CH3.gga_eng, "cphf_tol": 1e-10})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/ncdft_derivonce_uhf_ub3lyp.dat"), "rb") as f:
            ref_grad = pickle.load(f)["grad"].reshape(-1, 3)
        assert (np.allclose(helper.E_1, ref_grad, atol=1e-5, rtol=1e-4))
