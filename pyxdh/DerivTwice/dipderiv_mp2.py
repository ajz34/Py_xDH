import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceMP2, DerivTwiceXDH, DipDerivSCF, DipDerivNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipDerivMP2(DerivTwiceMP2, DipDerivSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + np.einsum("Bpq, Apq -> AB", self.pdB_D_r_oovv, A.B_1)
            + np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            + np.einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class DipDerivXDH(DerivTwiceXDH, DipDerivMP2, DipDerivNCDFT):

    def _get_E_2_U(self):
        return DipDerivMP2._get_E_2_U(self)


class Test_DipDerivMP2:

    def test_MP2_dipderiv(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2, GradMP2

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng, "cphf_tol": 1e-10}
        dip_helper = DipoleMP2(config)
        grad_helper = GradMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": grad_helper}
        helper = DipDerivMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-MP2-freq.fchk"))
        assert(np.allclose(E_2.T, formchk.dipolederiv(), atol=1e-6, rtol=1e-4))

    def test_B2PLYP_dipderiv(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2, GradMP2

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        grids_cphf = H2O2.gen_grids(50, 194)
        config = {"scf_eng": H2O2.gga_eng, "cc": 0.27, "cphf_grids": grids_cphf}
        dip_helper = DipoleMP2(config)
        grad_helper = GradMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": grad_helper}
        helper = DipDerivMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))
        assert(np.allclose(E_2.T, formchk.dipolederiv(), atol=1e-6, rtol=1e-4))

    def test_XYG3_dipderiv(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.DerivOnce import DipoleXDH, GradXDH
        import pickle

        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        grids_cphf = H2O2_sc.gen_grids(50, 194)
        config = {
            "scf_eng": H2O2_sc.gga_eng,
            "nc_eng": H2O2_nc.gga_eng,
            "cc": 0.3211,
            "cphf_grids": grids_cphf
        }
        dip_helper = DipoleXDH(config)
        grad_helper = GradXDH(config)
        config = {
            "deriv_A": dip_helper,
            "deriv_B": grad_helper,
        }

        helper = DipDerivXDH(config)
        E_2 = helper.E_2

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/xdh_dipderiv_xyg3.dat"), "rb") as f:
            ref_polar = pickle.load(f)["dipderiv"]
        assert(np.allclose(E_2.T, ref_polar, atol=1e-6, rtol=1e-4))
