import numpy as np

from pyxdh.DerivTwice import DerivTwiceUMP2, HessUSCF

# Cubic Inheritance: C2
class HessUMP2(DerivTwiceUMP2, HessUSCF):
    pass


class Test_HessUMP2:

    def test_UMP2_hess(self):
        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import GradUMP2

        CH3 = Mol_CH3()
        config = {
            "scf_eng": CH3.hf_eng,
            "cphf_tol": 1e-10,
        }
        grad_helper = GradUMP2(config)
        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
            "cphf_tol": 1e-10,
        }
        helper = HessUMP2(config)
        E_2 = helper.E_2
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-5, rtol=1e-4
        ))
