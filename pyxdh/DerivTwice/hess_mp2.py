import numpy as np

from pyxdh.DerivTwice import DerivTwiceMP2, HessSCF


# Cubic Inheritance: C2
class HessMP2(DerivTwiceMP2, HessSCF):
    pass


class Test_HessMP2:

    def test_MP2_hess(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import GradMP2

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.hf_eng,
            "rotation": True,
        }
        grad_helper = GradMP2(config)
        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
        }

        helper = HessMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-MP2-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B2PLYP_hess(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import GradMP2
        import pickle

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        config = {
            "scf_eng": H2O2.gga_eng,
            "cc": 0.27
        }
        grad_helper = GradMP2(config)

        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
        }

        helper = HessMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-5, rtol=1e-4
        ))

        with open(resource_filename("pyxdh", "Validation/numerical_deriv/mp2_hessian_b2plyp.dat"), "rb") as f:
            ref_hess = pickle.load(f)["hess"]

        assert (np.allclose(
            E_2, ref_hess,
            atol=1e-6, rtol=1e-4
        ))
