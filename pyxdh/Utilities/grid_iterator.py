from pyscf import dft, gto
import pyscf.dft.numint
from pyscf.dft import xcfun
import numpy as np
from functools import partial
import os

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GridIterator:

    def __init__(self, mol, grids, D, deriv=3, memory=2000):

        self.mol = mol  # type: gto.Mole
        self.grids = grids  # type: dft.Grids
        self.D = D
        self.ni = dft.numint.NumInt()
        self.ni.libxc = xcfun
        self.batch = self.ni.block_loop(mol, grids, mol.nao, deriv, memory)

        self._ao = None
        self._ngrid = None
        self._weight = None
        self._ao_0 = None
        self._ao_1 = None
        self._ao_2 = None
        self._ao_2T = None
        self._ao_3 = None
        self._ao_3T = None
        self._rho_01 = None
        self._rho_0 = None
        self._rho_1 = None
        self._rho_2 = None
        self._A_rho_1 = None
        self._A_rho_2 = None
        self._A_gamma_1 = None
        self._AB_rho_2 = None
        self._AB_rho_3 = None
        self._AB_gamma_2 = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.clear()
            self._ao, _, self._weight, _ = next(self.batch)
            return self
        except StopIteration:
            raise StopIteration

    def clear(self):
        self._ao = None
        self._ngrid = None
        self._weight = None
        self._ao_0 = None
        self._ao_1 = None
        self._ao_2 = None
        self._ao_2T = None
        self._ao_3 = None
        self._ao_3T = None
        self._rho_01 = None
        self._rho_0 = None
        self._rho_1 = None
        self._rho_2 = None
        self._A_rho_1 = None
        self._A_rho_2 = None
        self._A_gamma_1 = None
        self._AB_rho_2 = None
        self._AB_rho_3 = None
        self._AB_gamma_2 = None
        return

    # Property definition

    @property
    def ngrid(self):
        return self.weight.size

    @property
    def weight(self):
        return self._weight

    @property
    def ao(self):
        return self._ao

    @property
    def ao_0(self):
        if self._ao_0 is None:
            self._ao_0 = self.ao[0]
        return self._ao_0

    @property
    def ao_1(self):
        if self._ao_1 is None:
            self._ao_1 = self.ao[1:4]
        return self._ao_1

    @property
    def ao_2T(self):
        if self._ao_2T is None:
            self._ao_2T = self.ao[4:10]
        return self._ao_2T

    @property
    def ao_2(self):
        if self._ao_2 is None:
            XX, XY, XZ, YY, YZ, ZZ = range(4, 10)
            ao = self.ao
            self._ao_2 = np.array([
                [ao[XX], ao[XY], ao[XZ]],
                [ao[XY], ao[YY], ao[YZ]],
                [ao[XZ], ao[YZ], ao[ZZ]],
            ])
        return self._ao_2

    @property
    def ao_3(self):
        if self._ao_3 is None:
            XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
            ao = self.ao
            self._ao_3 = np.array([
                [[ao[XXX], ao[XXY], ao[XXZ]],
                 [ao[XXY], ao[XYY], ao[XYZ]],
                 [ao[XXZ], ao[XYZ], ao[XZZ]]],
                [[ao[XXY], ao[XYY], ao[XYZ]],
                 [ao[XYY], ao[YYY], ao[YYZ]],
                 [ao[XYZ], ao[YYZ], ao[YZZ]]],
                [[ao[XXZ], ao[XYZ], ao[XZZ]],
                 [ao[XYZ], ao[YYZ], ao[YZZ]],
                 [ao[XZZ], ao[YZZ], ao[ZZZ]]],
            ])
        return self._ao_3

    @property
    def ao_3T(self):
        if self._ao_3T is None:
            XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
            ao = self.ao
            self._ao_3T = np.array([
                [ao[XXX], ao[XXY], ao[XXZ], ao[XYY], ao[XYZ], ao[XZZ]],
                [ao[XXY], ao[XYY], ao[XYZ], ao[YYY], ao[YYZ], ao[YZZ]],
                [ao[XXZ], ao[XYZ], ao[XZZ], ao[YYZ], ao[YZZ], ao[ZZZ]],
            ])
        return self._ao_3T

    @property
    def rho_01(self):
        if self._rho_01 is None:
            self._rho_01 = np.zeros((4, self.ngrid))
            self._rho_01[0] = self.rho_0
            self._rho_01[1:4] = self.rho_1
        return self._rho_01

    @property
    def rho_0(self):
        if self._rho_0 is None:
            self._rho_0 = self.get_rho_0()
        return self._rho_0

    @property
    def rho_1(self):
        if self._rho_1 is None:
            self._rho_1 = self.get_rho_1()
        return self._rho_1

    @property
    def rho_2(self):
        if self._rho_2 is None:
            self._rho_2 = self.get_rho_2()
        return self._rho_2

    @property
    def A_rho_1(self):
        if self._A_rho_1 is None:
            self._A_rho_1 = self.get_A_rho_1()
        return self._A_rho_1

    @property
    def A_rho_2(self):
        if self._A_rho_2 is None:
            self._A_rho_2 = self.get_A_rho_2()
        return self._A_rho_2

    @property
    def A_gamma_1(self):
        if self._A_gamma_1 is None:
            self._A_gamma_1 = self.get_A_gamma_1()
        return self._A_gamma_1

    @property
    def AB_rho_2(self):
        if self._AB_rho_2 is None:
            self._AB_rho_2 = self.get_AB_rho_2()
        return self._AB_rho_2

    @property
    def AB_rho_3(self):
        if self._AB_rho_3 is None:
            self._AB_rho_3 = self.get_AB_rho_3()
        return self._AB_rho_3

    @property
    def AB_gamma_2(self):
        if self._AB_gamma_2 is None:
            self._AB_gamma_2 = self.get_AB_gamma_2()
        return self._AB_gamma_2

    # Function definition

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    def get_rho_0(self, D=None):
        """
        Generate density grid form generalized density matrix.

        .. math::

            \\rho = D_{\\mu \\nu} \\phi_\\mu \\phi_\\nu

        Parameters
        ----------
        D: np.ndarray or None

        Returns
        -------
        np.ndarray
        """
        if D is None:
            D = self.D
        rho_0 = np.einsum("uv, gu, gv -> g", D, self.ao_0, self.ao_0)
        return rho_0

    def get_rho_1(self, D=None):
        """
        Generate first order density derivative grid form generalized density matrix.

        .. math::

            \\rho_r = 2 D_{\\mu \\nu} \\phi_{r \\mu} \\phi_\\nu

        Parameters
        ----------
        D: np.ndarray or None

        Returns
        -------
        np.ndarray
        """
        if D is None:
            D = self.D
        rho_1 = 2 * np.einsum("uv, rgu, gv -> rg", D, self.ao_1, self.ao_0)
        return rho_1

    def get_rho_2(self, D=None):
        """
        Generate second order density derivative grid form generalized density matrix.

        .. math::

            \\rho_{rw} = 2 D_{\\mu \\nu} (\\phi_{rw \\mu} \\phi_\\nu + \\phi_{r \\mu} \\phi_{w \\nu})

        Parameters
        ----------
        D: np.ndarray or None

        Returns
        -------
        np.ndarray
        """
        if D is None:
            D = self.D
        rho_2 = (
            + 2 * np.einsum("uv, rwgu, gv -> rwg", D, self.ao_2, self.ao_0)
            + 2 * np.einsum("uv, rgu, wgv -> rwg", D, self.ao_1, self.ao_1)
        )
        return rho_2

    def get_A_rho_1(self, D=None):
        if D is None:
            D = self.D
        natm = self.mol.natm
        A_rho_1 = np.zeros((natm, 3, self.ngrid))
        for A in range(natm):
            sA = self.mol_slice(A)
            A_rho_1[A] = - 2 * np.einsum("tgk, gl, kl -> tg ", self.ao_1[:, :, sA], self.ao_0, D[sA])
        return A_rho_1

    def get_A_rho_2(self, D=None):
        if D is None:
            D = self.D
        natm = self.mol.natm
        A_rho_2 = np.zeros((natm, 3, 3, self.ngrid))
        for A in range(natm):
            sA = self.mol_slice(A)
            A_rho_2[A] = - 2 * np.einsum("trgk, gl, kl -> trg", self.ao_2[:, :, :, sA], self.ao_0, D[sA])
            A_rho_2[A] += - 2 * np.einsum("tgk, rgl, kl -> trg", self.ao_1[:, :, sA], self.ao_1, D[sA])
        return A_rho_2

    def get_A_gamma_1(self):
        A_gamma_1 = 2 * np.einsum("rg, Atrg -> Atg", self.rho_1, self.A_rho_2)
        return A_gamma_1

    def get_AB_rho_2(self, D=None):
        if D is None:
            D = self.D
        mol = self.mol
        natm = mol.natm
        AB_rho_2 = np.zeros((natm, natm, 3, 3, self.ngrid))
        for A in range(natm):
            sA = self.mol_slice(A)
            AB_rho_2[A, A] += 2 * np.einsum("tsgu, gv, uv -> tsg", self.ao_2[:, :, :, sA], self.ao_0, D[sA])
            for B in range(A + 1):
                sB = self.mol_slice(B)
                AB_rho_2[A, B] += 2 * np.einsum("tgu, sgv, uv -> tsg",
                                                self.ao_1[:, :, sA], self.ao_1[:, :, sB], D[sA, sB])
                if A != B:
                    AB_rho_2[B, A] = AB_rho_2[A, B].swapaxes(0, 1)
        return AB_rho_2

    def get_AB_rho_3(self, D=None):
        if D is None:
            D = self.D
        mol = self.mol
        natm = mol.natm
        AB_rho_3 = np.zeros((natm, natm, 3, 3, 3, self.ngrid))
        for A in range(natm):
            sA = self.mol_slice(A)
            AB_rho_3[A, A] += 2 * np.einsum("tsgu, rgv, uv -> tsrg", self.ao_2[:, :, :, sA], self.ao_1, D[sA])
            AB_rho_3[A, A] += 2 * np.einsum("tsrgu, gv, uv -> tsrg", self.ao_3[:, :, :, :, sA], self.ao_0, D[sA])
            for B in range(A + 1):
                _, _, p0B, p1B = mol.aoslice_by_atom()[B]
                sB = slice(p0B, p1B)
                AB_rho_3[A, B] += 2 * np.einsum("tgu, srgv, uv -> tsrg",
                                                self.ao_1[:, :, sA], self.ao_2[:, :, :, sB], D[sA, sB])
                AB_rho_3[A, B] += 2 * np.einsum("trgu, sgv, uv -> tsrg",
                                                self.ao_2[:, :, :, sA], self.ao_1[:, :, sB], D[sA, sB])
                if A != B:
                    AB_rho_3[B, A] = AB_rho_3[A, B].swapaxes(0, 1)
        return AB_rho_3

    def get_AB_gamma_2(self):
        AB_gamma_2 = (
            + 2 * np.einsum("Atrg, Bsrg -> ABtsg", self.A_rho_2, self.A_rho_2)
            + 2 * np.einsum("rg, ABtsrg -> ABtsg", self.rho_1, self.AB_rho_3)
        )
        return AB_gamma_2


class Test_GridIterator:

    def test_accordance_with_GridHelper(self):

        from pyxdh.Utilities.grid_helper import GridHelper

        mol = gto.Mole()
        mol.atom = """
        O  0.0  0.0  0.0
        H  1.5  0.0  0.0
        H  0.0  0.0  1.5
        """
        mol.basis = "6-31G"
        mol.verbose = 0
        mol.build()

        nao = mol.nao

        grids = dft.gen_grid.Grids(mol)
        grids.atom_grid = (75, 302)
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()

        dmX = np.random.random((nao, nao))
        dmX += dmX.T
        grdh = GridHelper(mol, grids, dmX)
        grdit = GridIterator(mol, grids, dmX, deriv=3, memory=100)

        # Should be able to save all grids in memory
        idx = 0
        for grdi in grdit:
            inc = grdi.rho_0.shape[0]
            s = slice(idx, idx + inc)
            assert(np.allclose(grdh.rho_0[s], grdi.rho_0))
            assert(np.allclose(grdh.rho_1[:, s], grdi.rho_1))
            assert(np.allclose(grdh.rho_2[:, :, s], grdi.rho_2))
            assert(np.allclose(grdh.A_rho_1[:, :, s], grdi.A_rho_1))
            assert(np.allclose(grdh.A_rho_2[:, :, :, s], grdi.A_rho_2))
            assert(np.allclose(grdh.AB_rho_2[:, :, :, :, s], grdi.AB_rho_2))
            assert(np.allclose(grdh.AB_rho_3[:, :, :, :, :, s], grdi.AB_rho_3))
            assert(np.allclose(grdh.A_gamma_1[:, :, s], grdi.A_gamma_1))
            assert(np.allclose(grdh.AB_gamma_2[:, :, :, :, s], grdi.AB_gamma_2))
            idx += inc
