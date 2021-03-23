import numpy as np
from pyscf import scf
from pyxdh.Utilities import NucCoordDerivGenerator, DipoleDerivGenerator, NumericDiff


class TestNumDeriv:

    def test_NucCoordDerivGenerator_by_SCFgrad(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2 = Mol_H2O2()
        mol = H2O2.mol
        hf_grad = H2O2.hf_grad

        generator = NucCoordDerivGenerator(mol, lambda mol_: scf.RHF(mol_).run())
        diff = NumericDiff(generator, lambda mf: mf.kernel())
        assert(np.allclose(
            hf_grad.kernel(),
            diff.derivative.reshape(mol.natm, 3),
            atol=1e-6, rtol=1e-4
        ))

        generator = NucCoordDerivGenerator(mol, lambda mol_: scf.RHF(mol_).run(), stencil=5)
        diff = NumericDiff(generator, lambda mf: mf.kernel())
        assert(np.allclose(
            hf_grad.kernel(),
            diff.derivative.reshape(mol.natm, 3),
            atol=1e-6, rtol=1e-4
        ))

    def test_DipoleDerivGenerator_by_SCF(self):
        from pyxdh.Utilities.test_molecules import Mol_H2O2
        H2O2 = Mol_H2O2()
        mol = H2O2.mol
        hf_eng = H2O2.hf_eng
        hf_eng.kernel()

        def mf_func(t, interval):
            mf = scf.RHF(mol)
            mf.get_hcore = lambda mol_: scf.rhf.get_hcore(mol_) - interval * mol_.intor("int1e_r")[t]
            return mf.kernel()

        generator = DipoleDerivGenerator(mf_func)
        diff = NumericDiff(generator)
        dip_nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
        assert(np.allclose(
            diff.derivative + dip_nuc,
            hf_eng.dip_moment(unit="A.U."),
            atol=1e-6, rtol=1e-4
        ))

        generator = DipoleDerivGenerator(mf_func, stencil=5)
        diff = NumericDiff(generator)
        assert(np.allclose(
            diff.derivative + dip_nuc,
            hf_eng.dip_moment(unit="A.U."),
            atol=1e-6, rtol=1e-4
        ))
