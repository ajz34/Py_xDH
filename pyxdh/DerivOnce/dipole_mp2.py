import numpy as np
from functools import partial
import os

from pyxdh.DerivOnce import DerivOnceMP2, DerivOnceXDH, DipoleSCF, DipoleNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipoleMP2(DerivOnceMP2, DipoleSCF):

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += super(DipoleMP2, self)._get_E_1()
        return E_1


class DipoleXDH(DerivOnceXDH, DipoleMP2, DipoleNCDFT):

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_DipoleMP2:

    @staticmethod
    def valid_assert(helper, resource_path):
        from pkg_resources import resource_filename
        from pyxdh.Utilities import FormchkInterface
        formchk = FormchkInterface(resource_filename("pyxdh", resource_path))
        assert(np.allclose(helper.E_1, formchk.dipole(), atol=1e-6, rtol=1e-4))

    def test_dipole_mp2(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2 = Mol_H2O2()
        config = {"scf_eng": H2O2.hf_eng}
        helper = DipoleMP2(config)
        self.valid_assert(helper, "Validation/gaussian/H2O2-MP2-freq.fchk")

    def test_dipole_b2plyp(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        grids_cphf = H2O2.gen_grids(50, 194)
        config = {"scf_eng": H2O2.gga_eng, "cc": 0.27, "cphf_grids": grids_cphf}
        helper = DipoleMP2(config)
        self.valid_assert(helper, "Validation/gaussian/H2O2-B2PLYP-freq.fchk")

    def test_dipole_xyg3(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        grids_cphf = H2O2_sc.gen_grids(50, 194)
        config = {"scf_eng": H2O2_sc.gga_eng, "nc_eng": H2O2_nc.gga_eng, "cc": 0.3211, "cphf_grids": grids_cphf}
        helper = DipoleXDH(config)
        self.valid_assert(helper, "Validation/gaussian/H2O2-XYG3-force.fchk")

    def test_dipole_xygjos(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP")
        grids_cphf = H2O2_sc.gen_grids(50, 194)
        config = {"scf_eng": H2O2_sc.gga_eng, "nc_eng": H2O2_nc.gga_eng, "cc": 0.4364, "ss": 0., "cphf_grids": grids_cphf}
        helper = DipoleXDH(config)
        self.valid_assert(helper, "Validation/gaussian/H2O2-XYGJOS-force.fchk")
