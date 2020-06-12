import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce import DerivOnceUMP2, GradUSCF, GradUNCDFT, DerivOnceUXDH

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradUMP2(DerivOnceUMP2, GradUSCF):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        D_r, B_1, W_I, S_1_mo, T_iajb, eri1_mo = self.D_r, self.B_1, self.W_I, self.S_1_mo, self.T_iajb, self.eri1_mo
        E_1 = (
            + np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
            + np.einsum("xpq, xApq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[0], eri1_mo[0][:, so[0], sv[0], so[0], sv[0]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[1], eri1_mo[1][:, so[0], sv[0], so[1], sv[1]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[2], eri1_mo[2][:, so[1], sv[1], so[1], sv[1]])
        ).reshape((natm, 3))
        E_1 += GradUSCF._get_E_1(self)
        return E_1


class GradUXDH(DerivOnceUXDH, GradUMP2, GradUNCDFT):

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        D_r, B_1, W_I, S_1_mo, T_iajb, eri1_mo = self.D_r, self.B_1, self.W_I, self.S_1_mo, self.T_iajb, self.eri1_mo
        E_1 = (
            + np.einsum("xpq, xApq -> A", self.D_r, self.B_1)
            + np.einsum("xpq, xApq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[0], eri1_mo[0][:, so[0], sv[0], so[0], sv[0]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[1], eri1_mo[1][:, so[0], sv[0], so[1], sv[1]])
            + 2 * np.einsum("iajb, Aiajb -> A", T_iajb[2], eri1_mo[2][:, so[1], sv[1], so[1], sv[1]])
        ).reshape((natm, 3))
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_GradUMP2:

    def test_UMP2_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3()
        helper = GradUMP2({"scf_eng": CH3.hf_eng.run(), "cphf_tol": 1e-10})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))

    def test_UB2PLYP_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3 = Mol_CH3(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        grids_cphf = CH3.gen_grids(atom_grid=(50, 194))
        helper = GradUMP2({"scf_eng": CH3.gga_eng.run(), "cphf_tol": 1e-10, "cc": 0.27, "cphf_grids": grids_cphf})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-B2PLYP-freq.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))

    def test_UXYG3_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3_sc = Mol_CH3(xc="B3LYPg")
        CH3_nc = Mol_CH3(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        grids_cphf = CH3_sc.gen_grids(atom_grid=(50, 194))
        helper = GradUXDH({
            "scf_eng": CH3_sc.gga_eng, "nc_eng": CH3_nc.gga_eng,
            "cc": 0.3211,
            "cphf_tol": 1e-10, "cphf_grids": grids_cphf})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-XYG3-force.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))

    def test_UXYGJOS_grad(self):

        from pyxdh.Utilities.test_molecules import Mol_CH3
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface

        CH3_sc = Mol_CH3(xc="B3LYPg")
        CH3_nc = Mol_CH3(xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP")
        grids_cphf = CH3_sc.gen_grids(atom_grid=(50, 194))
        helper = GradUXDH({"scf_eng": CH3_sc.gga_eng, "nc_eng": CH3_nc.gga_eng,
                           "cc": 0.4364, "ss": 0.,
                           "cphf_tol": 1e-10, "cphf_grids": grids_cphf})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-XYGJOS-force.fchk"))
        assert(np.allclose(helper.E_1, formchk.grad(), atol=1e-5, rtol=1e-4))
