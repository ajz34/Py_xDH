import numpy as np
from pyscf import gto, scf, lib


class AbstractDerivGenerator:

    def __init__(self):
        self.objects = NotImplemented  # type: np.ndarray
        self.stencil = NotImplemented  # type: int
        self.interval = NotImplemented  # type: float


class NucCoordDerivGenerator(AbstractDerivGenerator):

    def __init__(self, mol, mf_func, stencil=3, interval=3e-4):
        super(NucCoordDerivGenerator, self).__init__()
        self.mol = mol
        self.mf_func = mf_func
        self.objects = None
        self.stencil = stencil
        self.interval = interval / lib.param.BOHR
        self.init_objects()
        self.perform_mf()

    def init_objects(self):
        natm = self.mol.natm
        dim = natm * 3
        self.objects = np.empty((dim, self.stencil - 1), dtype=object)

    def move_mol(self, movelist):
        """
        Make a temporary molecule, which is moved away from the original molecule
        according to a pre-defined move list.

        Parameters
        ----------
        movelist : list of (tuple of int)
            The list is a series of movements on the temporary molecule.
            Every tuple in a list should contain 3 values. 1st value refers to
            atom index, 2nd value coordinate index, 3rd value times of interval
            to be moved away from original molecule.

        Returns
        -------
        mol_ret : pyscf.gto.Mole
        """
        # -------
        mol_ret = self.mol.copy()  # type: gto.Mole
        mol_ret_coords = mol_ret.atom_coords() * lib.param.BOHR
        for tup in movelist:
            mol_ret_coords[tup[0], tup[1]] += tup[2] * self.interval * lib.param.BOHR
        mol_ret.set_geom_(mol_ret_coords)
        return mol_ret

    def perform_mf(self):
        natm = self.mol.natm
        looplist = [(A, t, h)
                    for A in range(natm)
                    for t in range(3)
                    for h in range(self.stencil - 1)]
        if self.stencil == 5:
            dev_h = [-2, -1, 1, 2]
        else:
            dev_h = [-1, 1]
        for A, t, h in looplist:
            movelist = [(A, t, dev_h[h])]
            self.objects[3 * A + t, h] = self.mf_func(self.move_mol(movelist))


class NumericDiff(AbstractDerivGenerator):

    def __init__(self, scanner: AbstractDerivGenerator, num_method=None):
        super(NumericDiff, self).__init__()
        self.interval = scanner.interval
        self.stencil = scanner.stencil
        self.objects = scanner.objects
        self.num_method = num_method
        if self.num_method is None:
            self.num_method = lambda x: x
        self.num_matrix = NotImplemented  # type: np.ndarray
        self._derivative = NotImplemented  # type: np.ndarray

    @property
    def derivative(self):
        if self._derivative is not NotImplemented:
            return self._derivative
        # self.num_matrix = np.vectorize(self.num_method)(self.objects)
        self.num_matrix = np.empty_like(self.objects, dtype=object)
        for i in range(self.num_matrix.shape[0]):
            for j in range(self.num_matrix.shape[1]):
                self.num_matrix[i, j] = self.num_method(self.objects[i, j])
        self._derivative = []
        for matrices in self.num_matrix:
            if self.stencil == 3:
                self._derivative.append((matrices[1] - matrices[0]) / (2 * self.interval))
            elif self.stencil == 5:
                self._derivative.append((matrices[0] - 8 * matrices[1] + 8 * matrices[2] - matrices[3])
                                        / (12 * self.interval))
        self._derivative = np.array(self._derivative)
        return self._derivative


class DipoleDerivGenerator(AbstractDerivGenerator):

    def __init__(self, mf_func, stencil=3, interval=1e-6):
        super(DipoleDerivGenerator, self).__init__()
        self.mf_func = mf_func
        self.objects = NotImplemented
        self.stencil = stencil
        self.interval = interval
        self.init_objects()
        self.mf_func = mf_func
        self.perform_mf()

    def init_objects(self):
        self.objects = np.empty((3, self.stencil - 1), dtype=object)

    def perform_mf(self):
        looplist = [(t, h)
                    for t in range(3)
                    for h in range(self.stencil - 1)]
        if self.stencil == 5:
            dev_h = [-2, -1, 1, 2]
        else:
            dev_h = [-1, 1]
        for t, h in looplist:
            self.objects[t, h] = self.mf_func(t, dev_h[h] * self.interval)


class Test_DerivNumerical:

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
