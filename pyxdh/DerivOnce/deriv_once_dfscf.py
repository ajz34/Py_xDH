import numpy as np
from abc import ABC
from functools import partial
import os

from pyscf import gto

from pyxdh.DerivOnce import DerivOnceSCF

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceDFSCF(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceDFSCF, self).__init__(config)
        self.aux_jk = self.scf_eng.with_df.auxmol  # type: gto.Mole

    def _get_eri0_ao(self):
        raise AssertionError("eri0 should not be called in density fitting module!")

    def _get_eri1_ao(self):
        raise AssertionError("eri1 should not be called in density fitting module!")

