import numpy as np
from opt_einsum import contract as einsum

from pyxdh.DerivOnce import DerivOnceUSCF, DipoleSCF, DerivOnceUMP2
from pyxdh.Utilities import cached_property

from pyscf import gto, scf
from pyxdh.Utilities import FormchkInterface
from pkg_resources import resource_filename


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


class TestDipoleU:

    mol = gto.Mole(atom="C 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", spin=1, verbose=0).build()

    def test_u_uhf_dip(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol_grad = 1e-10; scf_eng.max_cycle = 128; scf_eng.run()
        diph = DipoleUSCF({"scf_eng": scf_eng})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-HF-freq.fchk"))
        # ASSERT: dipole - Gaussian
        assert np.allclose(diph.E_1, formchk.dipole(), atol=1e-6, rtol=1e-4)

    def test_u_mp2_dip(self):
        scf_eng = scf.UHF(self.mol); scf_eng.conv_tol_grad = 1e-12; scf_eng.max_cycle = 128; scf_eng.run()
        diph = DipoleUMP2({"scf_eng": scf_eng, "cphf_tol": 1e-12})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/CH3-MP2-freq.fchk"))
        # ASSERT: dipole - Gaussian
        assert np.allclose(diph.E_1, formchk.dipole(), atol=5e-6, rtol=2e-4)
