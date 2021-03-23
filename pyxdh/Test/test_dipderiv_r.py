import numpy as np
from pyscf import gto, scf, dft
from pyxdh.DerivOnce import GradSCF, GradMP2, GradXDH
from pyxdh.DerivOnce import DipoleSCF, DipoleMP2, DipoleXDH
from pyxdh.DerivTwice import DipDerivSCF, DipDerivMP2, DipDerivXDH
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface


class Test_DipDerivSCF:

    mol = gto.Mole(atom="N 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol); grids.atom_grid = (99, 590); grids.build()
    grids_cphf = dft.Grids(mol); grids_cphf.atom_grid = (50, 194); grids_cphf.build()

    def test_r_rhf_dipderiv(self):
        scf_eng = scf.RHF(self.mol).run()
        gradh = GradSCF({"scf_eng": scf_eng})
        diph = DipoleSCF({"scf_eng": scf_eng})
        ddh = DipDerivSCF({"deriv_A": diph, "deriv_B": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-HF-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(ddh.E_2.T, formchk.dipolederiv(), atol=5e-6, rtol=2e-4)

    def test_r_b3lyp_dipderiv(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        gradh = GradSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        diph = DipoleSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        ddh = DipDerivSCF({"deriv_A": diph, "deriv_B": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-B3LYP-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(ddh.E_2.T, formchk.dipolederiv(), atol=5e-6, rtol=2e-4)

    def test_r_mp2_dipderiv(self):
        scf_eng = scf.RHF(self.mol).run()
        gradh = GradMP2({"scf_eng": scf_eng})
        diph = DipoleMP2({"scf_eng": scf_eng})
        ddh = DipDerivMP2({"deriv_A": diph, "deriv_B": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-MP2-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(ddh.E_2.T, formchk.dipolederiv(), atol=5e-6, rtol=2e-4)

    def test_r_xyg3_dipderiv(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        diph = DipoleXDH(config)
        ddh = DipDerivXDH({"deriv_A": diph, "deriv_B": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYG3-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(ddh.E_2.T, formchk.dipolederiv(), atol=5e-6, rtol=2e-4)

    def test_r_xygjos_dipderiv(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        diph = DipoleXDH(config)
        ddh = DipDerivXDH({"deriv_A": diph, "deriv_B": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYGJOS-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(ddh.E_2.T, formchk.dipolederiv(), atol=5e-6, rtol=2e-4)
