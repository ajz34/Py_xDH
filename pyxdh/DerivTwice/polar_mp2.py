import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceMP2, DerivTwiceXDH, PolarSCF, PolarNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class PolarMP2(DerivTwiceMP2, PolarSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + np.einsum("Bpq, Apq -> AB", self.pdB_D_r_oovv, A.B_1)
            + np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            + np.einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class PolarXDH(DerivTwiceXDH, PolarMP2, PolarNCDFT):

    def _get_E_2_U(self):
        return PolarMP2._get_E_2_U(self)


class Test_PolarMP2:

    def test_MP2_polar(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng}
        dip_helper = DipoleMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": dip_helper}
        helper = PolarMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-MP2-freq.fchk"))
        assert(np.allclose(- E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4))

    def test_B2PLYP_polar(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        grids_cphf = H2O2.gen_grids(50, 194)
        config = {"scf_eng": H2O2.gga_eng, "cc": 0.27, "cphf_grids": grids_cphf}
        dip_helper = DipoleMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": dip_helper}
        helper = PolarMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))
        assert(np.allclose(- E_2, formchk.polarizability(), atol=1e-6, rtol=1e-4))

    def test_XYG3_polar(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.DerivOnce import DipoleXDH
        import pickle

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/xdh_polarizability_xyg3.dat"), "rb") as f:
            ref_polar = pickle.load(f)["polarizability"]

        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        grids_cphf = H2O2_sc.gen_grids(50, 194)
        dip_config = {
            "scf_eng": H2O2_sc.gga_eng,
            "nc_eng": H2O2_nc.gga_eng,
            "cc": 0.3211,
            "cphf_grids": grids_cphf
        }
        # With rotation
        dip_helper = DipoleXDH(dip_config)
        polar_config = {"deriv_A": dip_helper, "deriv_B": dip_helper}
        helper = PolarXDH(polar_config)
        assert(np.allclose(- helper.E_2, ref_polar, atol=1e-6, rtol=1e-4))
        # No rotation
        dip_config["rotation"] = False
        dip_helper = DipoleXDH(dip_config)
        polar_config = {"deriv_A": dip_helper, "deriv_B": dip_helper}
        helper = PolarXDH(polar_config)
        assert(np.allclose(- helper.E_2, ref_polar, atol=1e-6, rtol=1e-4))
