import numpy as np
from pyscf import gto, scf
from pyxdh.DerivOnce import DipoleUSCF, DipoleUMP2
from pyxdh.DerivTwice import PolarUSCF, PolarUMP2
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename


class TestPolarU:

    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()

    def test_u_uhf_polar(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
        diph = DipoleUSCF({"scf_eng": scf_eng})
        polh = PolarUSCF({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)

    def test_u_mp2_polar(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
        diph = DipoleUMP2({"scf_eng": scf_eng, "cphf_tol": 1e-12})
        polh = PolarUMP2({"deriv_A": diph})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        assert np.allclose(- polh.E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4)