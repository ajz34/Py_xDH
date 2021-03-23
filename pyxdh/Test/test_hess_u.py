import numpy as np
from pyscf import gto, scf
from pyxdh.DerivOnce import GradUSCF
from pyxdh.DerivTwice import HessUSCF
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename


class TestHessU:

    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()

    def test_u_uhf_hess(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10
        scf_eng.max_cycle = 256; scf_eng.run()
        gradh = GradUSCF({"scf_eng": scf_eng})
        hessh = HessUSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)

    # TODO: UMP2 Hessian is possibly flawed
    # def test_u_mp2_hess(self):
    #     scf_eng = scf.UHF(self.mol); scf_eng.conv_tol = 1e-12; scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 256; scf_eng.run()
    #     gradh = GradUMP2({"scf_eng": scf_eng, "cphf_tol": 1e-10})
    #     hessh = HessUMP2({"deriv_A": gradh})
    #     formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
    #     assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-5, rtol=1e-4)
