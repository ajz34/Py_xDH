import numpy as np
from functools import partial
import os

from pyscf import grad

from pyxdh.DerivOnce import DerivOnceUSCF, DerivOnceNCDFT, GradSCF
from pyxdh.Utilities import GridIterator, KernelHelper

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradUSCF(DerivOnceUSCF, GradSCF):

    def _get_F_1_ao(self):
        return np.array(self.scf_hess.make_h1(self.C, self.mo_occ)).reshape((2, self.natm * 3, self.nao, self.nao))

    def _get_E_1(self):
        mol = self.mol
        natm = self.natm
        cx, xc = self.cx, self.xc
        H_1_ao = self.H_1_ao
        eri1_ao = self.eri1_ao
        S_1_mo = self.S_1_mo
        F_0_mo = self.F_0_mo
        occ = self.occ
        D = self.D
        grids = self.grids
        grdit_memory = self.grdit_memory

        E_1 = (
                + np.einsum("Auv, xuv -> A", H_1_ao, D)
                + 0.5 * np.einsum("Auvkl, yuv, xkl -> A", eri1_ao, D, D)
                - 0.5 * cx * np.einsum("Aukvl, xuv, xkl -> A", eri1_ao, D, D)
                - np.einsum("xApq, xpq, xp, xq -> A", S_1_mo, F_0_mo, occ, occ)
                + grad.rhf.grad_nuc(mol).reshape(-1)
        )

        # GGA part contiribution
        if self.xc_type == "GGA":
            grdit = zip(GridIterator(mol, grids, D[0], deriv=2), GridIterator(mol, grids, D[1], deriv=2))
            for grdh in grdit:
                kerh = KernelHelper(grdh, xc)
                E_1 += (
                    + np.einsum("g, Atg -> At", kerh.fr[0], grdh[0].A_rho_1)
                    + np.einsum("g, Atg -> At", kerh.fr[1], grdh[1].A_rho_1)
                    + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg[0], grdh[0].rho_1, grdh[0].A_rho_2)
                    + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg[2], grdh[1].rho_1, grdh[1].A_rho_2)
                    + 1 * np.einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[1].rho_1, grdh[0].A_rho_2)
                    + 1 * np.einsum("g, rg, Atrg -> At", kerh.fg[1], grdh[0].rho_1, grdh[1].A_rho_2)
            ).reshape(-1)

        return E_1.reshape((natm, 3))


class Test_GradUSCF:

    def test_UHF_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = GradUSCF({"scf_eng": CH3.hf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))

    def test_UB3LYP_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = GradUSCF({"scf_eng": CH3.gga_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-B3LYP-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-6, rtol=1e-4))
