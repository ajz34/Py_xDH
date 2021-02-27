import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce import DerivOnceUMP2, DipoleUSCF, DerivOnceUXDH

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipoleUMP2(DerivOnceUMP2, DipoleUSCF):

    def _get_E_1(self):
        E_1 = np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
        E_1 += DipoleUSCF._get_E_1(self)
        return E_1


class Test_DipoleUMP2:

    def test_dipole_ump2(self):
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface
        from pyxdh.Utilities.test_molecules import Mol_CH3
        CH3 = Mol_CH3()
        config = {"scf_eng": CH3.hf_eng, "cphf_tol": 1e-10}
        helper = DipoleUMP2(config)
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        assert (np.allclose(helper.E_1, formchk.dipole(), atol=1e-5, rtol=1e-4))

