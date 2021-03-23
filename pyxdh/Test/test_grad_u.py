import numpy as np
from pyscf import gto, scf, dft, mp
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface
import pickle
from pyxdh.DerivOnce import GradUSCF, GradUNCDFT, GradUMP2, GradUXDH


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

    def test_u_xyg3_grad(self):
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

    def test_u_xygjos_grad(self):
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
