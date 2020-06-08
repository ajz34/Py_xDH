from pyscf import scf, gto, grad, dft


class Mol_H2O2:

    def __init__(self, xc="B3LYPg", mol=None):

        if mol is None:
            mol = gto.Mole()
            mol.atom = """
            O  0.0  0.0  0.0
            O  0.0  0.0  1.5
            H  1.0  0.0  0.0
            H  0.0  0.7  1.0
            """
            mol.basis = "6-31G"
            mol.verbose = 0
            mol.build()

        self.mol = mol
        self.xc = xc

        self._hf_eng = NotImplemented
        self._hf_grad = NotImplemented
        self._gga_eng = NotImplemented
        self._gga_grad = NotImplemented

    @property
    def hf_eng(self):
        if self._hf_eng is not NotImplemented:
            return self._hf_eng
        hf_eng = scf.RHF(self.mol)
        self._hf_eng = hf_eng
        return self._hf_eng

    @property
    def hf_grad(self):
        if self._hf_grad is not NotImplemented:
            return self._hf_grad
        if self.hf_eng.mo_coeff is None:
            self.hf_eng.kernel()
        hf_grad = grad.RHF(self.hf_eng)
        self._hf_grad = hf_grad
        return self._hf_grad

    @property
    def gga_eng(self):
        if self._gga_eng is not NotImplemented:
            return self._gga_eng

        grids = self.gen_grids()

        gga_eng = scf.RKS(self.mol)
        gga_eng.grids = grids
        gga_eng.conv_tol = 1e-11
        gga_eng.conv_tol_grad = 1e-9
        gga_eng.max_cycle = 100
        gga_eng.xc = self.xc

        self._gga_eng = gga_eng
        return self._gga_eng

    @property
    def gga_grad(self):
        if self._gga_grad is not NotImplemented:
            return self._gga_grad
        if self.gga_eng.mo_coeff is None:
            self.gga_eng.kernel()
        gga_grad = grad.RKS(self.gga_eng)
        self._gga_grad = gga_grad
        return self._gga_grad

    def gen_grids(self, rad_points=99, sph_points=590):
        grids = dft.Grids(self.mol)
        grids.atom_grid = (rad_points, sph_points)
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()
        return grids


class Mol_CH3:

    def __init__(self, xc="B3LYPg", mol=None, atom_grid=(99, 590)):

        if mol is None:
            mol = gto.Mole()
            mol.atom = """
            C  0. 0. 0.
            H  1. 0. 0.
            H  0. 2. 0.
            H  0. 0. 1.5
            """
            mol.basis = "6-31G"
            mol.spin = 1
            mol.verbose = 0
            mol.build()

        self.mol = mol
        self.xc = xc
        self.atom_grid = atom_grid

        self._hf_eng = NotImplemented
        self._hf_grad = NotImplemented
        self._gga_eng = NotImplemented
        self._gga_grad = NotImplemented

    @property
    def hf_eng(self):
        if self._hf_eng is not NotImplemented:
            return self._hf_eng
        hf_eng = scf.UHF(self.mol)
        self._hf_eng = hf_eng
        return self._hf_eng

    @property
    def hf_grad(self):
        if self._hf_grad is not NotImplemented:
            return self._hf_grad
        if self.hf_eng.mo_coeff is None:
            self.hf_eng.kernel()
        hf_grad = grad.UHF(self.hf_eng)
        self._hf_grad = hf_grad
        return self._hf_grad

    @property
    def gga_eng(self):
        if self._gga_eng is not NotImplemented:
            return self._gga_eng

        grids = self.gen_grids(atom_grid=self.atom_grid)

        gga_eng = scf.UKS(self.mol)
        gga_eng.grids = grids
        gga_eng.conv_tol = 1e-11
        gga_eng.conv_tol_grad = 1e-9
        gga_eng.max_cycle = 100
        gga_eng.xc = self.xc

        self._gga_eng = gga_eng
        return self._gga_eng

    @property
    def gga_grad(self):
        if self._gga_grad is not NotImplemented:
            return self._gga_grad
        if self.gga_eng.mo_coeff is None:
            self.gga_eng.kernel()
        gga_grad = grad.UKS(self.gga_eng)
        self._gga_grad = gga_grad
        return self._gga_grad

    def gen_grids(self, atom_grid=None):
        grids = dft.Grids(self.mol)
        grids.atom_grid = atom_grid
        if atom_grid is None:
            grids.atom_grid = (99, 590)
        grids.radi_method = dft.gauss_chebyshev
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()
        return grids
