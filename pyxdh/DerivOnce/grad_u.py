import numpy as np
from opt_einsum import contract as einsum

from pyscf import grad

from pyxdh.DerivOnce import DerivOnceUSCF, GradSCF, DerivOnceUNCDFT, DerivOnceUMP2, DerivOnceUXDH
from pyxdh.Utilities import GridIterator, KernelHelper, timing, cached_property

from pyscf import gto, scf, dft, mp
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename
import pickle


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


class TestGradU:

    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()
    grids = dft.Grids(mol); grids.atom_grid = (99, 590); grids.build()
    grids_cphf = dft.Grids(mol); grids_cphf.atom_grid = (50, 194); grids_cphf.build()

    def test_u_uhf_grad(self):
        scf_eng = scf.UHF(self.mol).run()
        scf_grad = scf_eng.Gradients().run()
        gradh = GradUSCF({"scf_eng": scf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        # ASSERT: energy - Gaussian
        assert np.allclose(gradh.eng, formchk.total_energy())
        # ASSERT: energy - PySCF
        assert np.allclose(gradh.eng, scf_eng.e_tot)
        # ASSERT: grad - Gaussian
        assert np.allclose(gradh.E_1, formchk.grad(), atol=5e-6, rtol=1e-4)
        # ASSERT: grad - PySCF
        assert np.allclose(gradh.E_1, scf_grad.de, atol=1e-6, rtol=1e-4)

    def test_u_b3lyp_grad(self):
        scf_eng = dft.UKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        scf_grad = scf_eng.Gradients().run()
        gradh = GradUSCF({"scf_eng": scf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-B3LYP-freq.fchk"))
        # ASSERT: energy - Gaussian
        assert np.allclose(gradh.eng, formchk.total_energy())
        # ASSERT: energy - PySCF
        assert np.allclose(gradh.eng, scf_eng.e_tot)
        # ASSERT: grad - Gaussian
        assert np.allclose(gradh.E_1, formchk.grad(), atol=5e-6, rtol=1e-4)
        # ASSERT: grad - PySCF
        assert np.allclose(gradh.E_1, scf_grad.de, atol=1e-6, rtol=1e-4)

    def test_u_hfb3lyp_grad(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol_grad = 1e-10; scf_eng.run()
        nc_eng = dft.UKS(self.mol, xc="B3LYPg")
        nc_eng.grids = self.grids
        gradh = GradUNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng, "cphf_tol": 1e-15})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/CH3-HFB3LYP-grad.dat"), "rb") as f:
            ref_grad = pickle.load(f).reshape(-1, 3)
        # ASSERT: energy - theoretical
        assert np.allclose(gradh.eng, nc_eng.energy_tot(dm=scf_eng.make_rdm1()))
        # ASSERT: grad - numerical
        assert np.allclose(gradh.E_1, ref_grad, atol=1e-6, rtol=1e-4)

    def test_u_mp2_grad(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 128; scf_eng.run()
        mp2_eng = mp.MP2(scf_eng).run()
        mp2_grad = mp2_eng.Gradients().run()
        gradh = GradUMP2({"scf_eng": scf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        # ASSERT: energy - Gaussian
        assert np.allclose(gradh.eng, formchk.total_energy())
        # ASSERT: energy - PySCF
        assert np.allclose(gradh.eng, mp2_eng.e_tot)
        # ASSERT: grad - Gaussian
        assert np.allclose(gradh.E_1, formchk.grad(), atol=1e-6, rtol=1e-4)
        # ASSERT: grad - PySCF
        assert np.allclose(gradh.E_1, mp2_grad.de, atol=1e-6, rtol=1e-4)

    def test_r_xyg3_grad(self):
        scf_eng = dft.UKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids
        scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 128; scf_eng.run()
        nc_eng = dft.UKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf, "cphf_tol": 1e-10}
        gradh = GradUXDH(config)
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-XYG3-force.fchk"))
        # ASSERT: energy - Gaussian
        assert np.allclose(gradh.eng, formchk.total_energy())
        # ASSERT: grad - Gaussian
        assert np.allclose(gradh.E_1, formchk.grad(), atol=5e-6, rtol=1e-4)

    def test_r_xygjos_grad(self):
        scf_eng = dft.UKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids
        scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 128; scf_eng.run()
        nc_eng = dft.UKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        gradh = GradUXDH(config)
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-XYGJOS-force.fchk"))
        # ASSERT: energy - Gaussian
        assert np.allclose(gradh.eng, formchk.total_energy())
        # ASSERT: grad - Gaussian
        assert np.allclose(gradh.E_1, formchk.grad(), atol=5e-6, rtol=1e-4)
