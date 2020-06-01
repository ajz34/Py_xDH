import numpy as np
from abc import ABC, abstractmethod
from functools import partial
import os
import warnings
import copy
from pyscf.scf._response_functions import _gen_rhf_response

from pyscf import gto, dft, grad, hessian
from pyscf.scf import cphf

from pyxdh.Utilities import timing
from pyxdh.DerivOnce.deriv_once_scf import DerivOnceSCF
from pyxdh.Utilities import GridIterator, KernelHelper, timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceUSCF(DerivOnceSCF, ABC):

    def initialization_scf(self):
        if self.init_scf:
            self.mo_occ = self.scf_eng.mo_occ
            self.C = self.scf_eng.mo_coeff
            self.e = self.scf_eng.mo_energy
            self.nocc = self.mol.nelec
        return

    def initialization_pyscf(self):
        if (self.scf_eng.mo_coeff is NotImplemented or self.scf_eng.mo_coeff is None) and self.init_scf:
            self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        if isinstance(self.scf_eng, dft.rks.RKS):
            self.xc = self.scf_eng.xc
            self.grids = self.scf_eng.grids
            self.xc_type = dft.libxc.xc_type(self.xc)
            self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
            self.scf_grad = grad.uks.Gradients(self.scf_eng)
            self.scf_hess = hessian.uks.Hessian(self.scf_eng)
        else:
            self.scf_grad = grad.UHF(self.scf_eng)
            self.scf_hess = hessian.UHF(self.scf_eng)
        return

    @property
    def so(self):
        return slice(0, self.nocc[0]), slice(0, self.nocc[1])

    @property
    def sv(self):
        return slice(self.nocc[0], self.nmo), slice(self.nocc[1], self.nmo)

    @property
    def sa(self):
        return slice(0, self.nmo), slice(0, self.nmo)

    @property
    def Co(self):
        return self.C[0, :, self.so[0]], self.C[0, :, self.so[1]]

    @property
    def Cv(self):
        return self.Cv[0, :, self.so[0]], self.Cv[0, :, self.so[1]]

    @property
    def eo(self):
        return self.e[0, self.so[0]], self.e[0, self.so[1]]

    @property
    def ev(self):
        return self.e[0, self.sv[0]], self.e[0, self.sv[1]]

    def _get_D(self):
        return np.einsum("xup, xp, xvp -> xuv", self.C, self.occ, self.C)

    def _get_H_0_mo(self):
        return np.einsum("xup, uv, xvq -> xpq", self.C, self.H_0_ao, self.C)

    def _get_S_0_mo(self):
        return np.einsum("xup, uv, xvq -> xpq", self.C, self.S_0_ao, self.C)

    def _get_F_0_mo(self):
        return np.einsum("xup, xuv, xvq -> xpq", self.C, self.F_0_ao, self.C)

    def _get_H_1_mo(self):
        if not isinstance(self.H_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, xup, xvq -> xApq", self.H_1_ao, self.C, self.C)

    def _get_S_1_mo(self):
        if not isinstance(self.S_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, xup, xvq -> xApq", self.S_1_ao, self.C, self.C)

    def _get_F_1_mo(self):
        if not isinstance(self.F_1_ao, np.ndarray):
            return 0
        return np.einsum("xAuv, xup, xvq -> xApq", self.F_1_ao, self.C, self.C)




