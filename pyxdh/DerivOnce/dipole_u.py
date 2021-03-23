import numpy as np
from opt_einsum import contract as einsum

from pyxdh.DerivOnce import DerivOnceUSCF, DipoleSCF, DerivOnceUMP2
from pyxdh.Utilities import cached_property


class DipoleUSCF(DerivOnceUSCF, DipoleSCF):

    @cached_property
    def F_1_ao(self):
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

        dip_elec = einsum("Apq, xpq -> A", H_1_ao, D)
        dip_nuc = einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())

        return dip_elec + dip_nuc


class DipoleUMP2(DerivOnceUMP2, DipoleUSCF):

    @cached_property
    def eri1_mo(self):
        return 0

    def _get_E_1(self):
        E_1 = np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
        E_1 += DipoleUSCF._get_E_1(self)
        return E_1
