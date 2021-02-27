import numpy as np

from pyxdh.DerivTwice import DerivTwiceUMP2, PolarUSCF

# Cubic Inheritance: C2
class PolarUMP2(DerivTwiceUMP2, PolarUSCF):

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            + np.einsum("xBpq, xApq -> AB", self.pdB_D_r_oovv, A.B_1)
            + np.einsum("Aai, Bai -> AB", A.U_1[0, :, sv[0], so[0]], self.RHS_B[0])
            + np.einsum("Aai, Bai -> AB", A.U_1[1, :, sv[1], so[1]], self.RHS_B[1])
            + np.einsum("xpq, xABpq -> AB", self.D_r, self.pdB_F_A_mo)
        )
        return E_2_MP2_Contrib


class Test_PolarUMP2:

    def test_UMP2_polar(self):
        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleUMP2

        CH3 = Mol_CH3()
        config = {
            "scf_eng": CH3.hf_eng,
            "cphf_tol": 1e-10,
        }
        grad_helper = DipoleUMP2(config)
        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
            "cphf_tol": 1e-10,
        }
        helper = PolarUMP2(config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))

        assert(np.allclose(
            -E_2, formchk.polarizability(),
            atol=1e-6, rtol=1e-4
        ))
