import numpy as np
from pyscf import gto, scf, dft
from pyxdh.DerivOnce import DipoleSCF, DipoleNCDFT, DipoleMP2, DipoleXDH
from pyxdh.DerivTwice import PolarSCF, PolarNCDFT, PolarMP2, PolarXDH
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface
import pickle


class TestPolarR:

    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol); grids.atom_grid = (99, 590); grids.build()
    grids_cphf = dft.Grids(mol); grids_cphf.atom_grid = (50, 194); grids_cphf.build()

    def test_r_rhf_polar(self):
        scf_eng = scf.RHF(self.mol).run()
        diph = DipoleSCF({"scf_eng": scf_eng})
        polh = PolarSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-HF-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_b3lyp_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        diph = DipoleSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        polh = PolarSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-B3LYP-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_hfb3lyp_polar(self):
        scf_eng = scf.RHF(self.mol); scf_eng.conv_tol_grad = 1e-8; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="B3LYPg")
        nc_eng.grids = self.grids
        diph = DipoleNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng})
        polh = PolarNCDFT({"deriv_A": diph})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/NH3-HFB3LYP-pol.dat"), "rb") as f:
            ref_polar = pickle.load(f)
        # ASSERT: polar - numerical
        assert np.allclose(- polh.E_2, ref_polar, atol=1e-6, rtol=1e-4)

    def test_r_mp2_polar(self):
        scf_eng = scf.RHF(self.mol).run()
        diph = DipoleMP2({"scf_eng": scf_eng})
        polh = PolarMP2({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-MP2-freq.fchk"))
        # ASSERT: polar - Gaussian
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_xyg3_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf}
        diph = DipoleXDH(config)
        polh = PolarXDH({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYG3-freq.fchk"))
        # ASSERT: polar - Gaussian
        np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_r_xygjos_polar(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        diph = DipoleXDH(config)
        polh = PolarXDH({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYGJOS-freq.fchk"))
        # ASSERT: polar - Gaussian
        np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)