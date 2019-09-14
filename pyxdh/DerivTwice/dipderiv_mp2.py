import numpy as np
from functools import partial
import os

from pyxdh.DerivTwice import DerivTwiceMP2, DipDerivSCF

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


class Test_DipDerivMP2:

    def test_MP2_dipderiv(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2, GradMP2

        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng}
        dip_helper = DipoleMP2(config)
        grad_helper = GradMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": grad_helper}
        helper = DipDerivMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-MP2-freq.fchk"))
        assert(np.allclose(E_2.T, formchk.dipolederiv(), atol=1e-6, rtol=1e-4))

    def test_B2PLYP_polar(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.DerivOnce import DipoleMP2, GradMP2

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        config = {"scf_eng": H2O2.gga_eng, "cc": 0.27}
        dip_helper = DipoleMP2(config)
        grad_helper = GradMP2(config)
        config = {"deriv_A": dip_helper, "deriv_B": grad_helper}
        helper = DipDerivMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))
        assert(np.allclose(E_2.T, formchk.dipolederiv(), atol=1e-6, rtol=1e-4))
