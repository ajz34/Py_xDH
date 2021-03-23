import numpy as np
from pyscf import gto, scf, dft
from pyxdh.DerivOnce import GradSCF, GradNCDFT, GradMP2, GradXDH
from pyxdh.DerivTwice import HessSCF, HessNCDFT, HessMP2, HessXDH
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface
import pickle


class TestHessR:

    mol = gto.Mole(atom="N 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol); grids.atom_grid = (99, 590); grids.build()
    grids_cphf = dft.Grids(mol); grids_cphf.atom_grid = (50, 194); grids_cphf.build()

    def test_r_rhf_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        scf_hess = scf_eng.Hessian().run()
        gradh = GradSCF({"scf_eng": scf_eng})
        hessh = HessSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-HF-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)
        # ASSERT: hessian - PySCF
        assert np.allclose(hessh.E_2, scf_hess.de.swapaxes(-2, -3).reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_b3lyp_hess(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        scf_hess = scf_eng.Hessian().run()
        gradh = GradSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        hessh = HessSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-B3LYP-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-5, rtol=2e-4)
        # ASSERT: hessian - PySCF
        assert np.allclose(hessh.E_2, scf_hess.de.swapaxes(-2, -3).reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_hfb3lyp_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        nc_eng = dft.RKS(self.mol, xc="B3LYPg")
        nc_eng.grids = self.grids
        gradh = GradNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng})
        hessh = HessNCDFT({"deriv_A": gradh})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/NH3-HFB3LYP-hess.dat"), "rb") as f:
            ref_hess = pickle.load(f)
        # ASSERT: hessian - numerical
        assert np.allclose(hessh.E_2, ref_hess.reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_mp2_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        gradh = GradMP2({"scf_eng": scf_eng})
        hessh = HessMP2({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-MP2-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)

    def test_r_xyg3_hess(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        hessh = HessXDH({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYG3-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(hessh.E_2, formchk.hessian(), atol=2e-5, rtol=2e-4)

    def test_r_xygjos_grad(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        hessh = HessXDH({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYGJOS-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(hessh.E_2, formchk.hessian(), atol=2e-5, rtol=2e-4)
