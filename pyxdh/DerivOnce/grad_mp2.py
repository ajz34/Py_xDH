import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce import DerivOnceMP2, DerivOnceXDH, GradSCF, GradNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: C2
class GradMP2(DerivOnceMP2, GradSCF):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        E_1 = (
            + np.einsum("pq, Apq -> A", self.D_r, self.B_1)
            + np.einsum("pq, Apq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", self.T_iajb, self.eri1_mo[:, so, sv, so, sv])
        ).reshape(natm, 3)
        E_1 += super(GradMP2, self)._get_E_1()
        return E_1


# Cubic Inheritance: D2
class GradXDH(DerivOnceXDH, GradMP2, GradNCDFT):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        E_1 = (
            + np.einsum("pq, Apq -> A", self.D_r, self.B_1)
            + np.einsum("pq, Apq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", self.T_iajb, self.eri1_mo[:, so, sv, so, sv])
        ).reshape(natm, 3)
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_GradMP2:

    def test_MP2_grad(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface

        from pyscf import mp, grad

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.hf_eng
        }
        gmh = GradMP2(config)

        mp2_eng = mp.MP2(gmh.scf_eng)
        mp2_eng.kernel()
        mp2_grad = grad.mp2.Gradients(mp2_eng)
        mp2_grad.kernel()

        assert(np.allclose(
            gmh.E_1, mp2_grad.de,
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-MP2-freq.fchk"))

        assert(np.allclose(
            gmh.E_1, formchk.grad(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B2PLYP_grad(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        config = {
            "scf_eng": H2O2.gga_eng,
            "cc": 0.27
        }
        gmh = GradMP2(config)

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))

        assert(np.allclose(
            gmh.eng,
            formchk.total_energy()
        ))
        assert(np.allclose(
            gmh.E_1, formchk.grad(),
            atol=1e-5, rtol=1e-4
        ))

    def test_XYG3_grad(self):

        from pkg_resources import resource_filename
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        from pyxdh.Utilities import FormchkInterface

        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        config = {
            "scf_eng": H2O2_sc.gga_eng,
            "nc_eng": H2O2_nc.gga_eng,
            "cc": 0.3211
        }
        gmh = GradXDH(config)

        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/H2O2-XYG3-force.fchk"))

        assert(np.allclose(
            gmh.eng,
            formchk.total_energy()
        ))
        assert(np.allclose(
            gmh.E_1, formchk.grad(),
            atol=1e-5, rtol=1e-4
        ))
