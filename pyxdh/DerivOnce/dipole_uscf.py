import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce import DerivOnceUSCF, DipoleSCF, DerivOnceUNCDFT
from pyxdh.Utilities import GridIterator, KernelHelper, timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipoleUSCF(DerivOnceUSCF, DipoleSCF):

    def _get_F_1_ao(self):
        return np.array([self.H_1_ao, self.H_1_ao])

    def Ax1_Core(self, si, sj, sk, sl, reshape=True):

        def fx(_):
            # Method is only used in DFT
            if self.xc_type == "HF":
                return 0
            else:
                raise NotImplementedError("DFT is not implemented!")

        return fx

    def _get_E_1(self):
        mol = self.mol
        H_1_ao = self.H_1_ao
        D = self.D

        dip_elec = np.einsum("Apq, xpq -> A", H_1_ao, D)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())

        return dip_elec + dip_nuc


class Test_GradUSCF:

    def test_UHF_dipole(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = DipoleUSCF({"scf_eng": CH3.hf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.dipole(), atol=1e-6, rtol=1e-4))
