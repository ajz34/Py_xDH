# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceUSCF, DerivTwiceUMP2, HessSCF
from pyxdh.Utilities import timing, cached_property


class HessUSCF(DerivTwiceUSCF, HessSCF):

    @cached_property
    @timing
    def F_2_ao_JKcontrib(self):
        D = self.D
        eri2_ao = self.eri2_ao
        return (
            einsum("ABuvkl, xkl -> ABuv", eri2_ao, D)[None, :].repeat(2, axis=0),
            einsum("ABukvl, xkl -> xABuv", eri2_ao, D),
        )

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        D = self.D
        cx = self.cx if cx is None else cx
        # HF Contribution
        E_SS_HF_contrib = (
            + einsum("ABuv, xuv -> AB", self.H_2_ao, D)
            + 0.5 * einsum("xABuv, xuv -> AB", self.F_2_ao_Jcontrib - cx * self.F_2_ao_Kcontrib, D)
        )
        return E_SS_HF_contrib


class HessUMP2(DerivTwiceUMP2, HessUSCF):
    pass


