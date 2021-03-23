from typing import Tuple

from pyscf import dft, gto
import pyscf.dft.numint
import numpy as np
from functools import partial
import os

from pyxdh.Utilities.grid_iterator import GridIterator

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GridHelperLegacy:

    def __init__(self, mol, grids, D, engine="xcfun"):
        # warnings.warn("GridHelper is considered memory consuming!")

        # Initialization Parameters
        self.mol = mol  # type: pyscf.gto.Mole
        self.grids = grids  # type: pyscf.dft.gen_grid.Grids
        self.D = D  # type: np.ndarray

        # Calculation
        nao = mol.nao
        ni = dft.numint.NumInt()
        if engine == "xcfun":
            from pyscf.dft import xcfun
            ni.libxc = xcfun
        ngrid = grids.weights.size
        grid_weight = grids.weights
        grid_ao = np.zeros((20, ngrid, nao))  # 20 at first dimension is related to 3rd derivative of orbital
        current_grid_count = 0
        for ao, _, _, _ in ni.block_loop(mol, grids, nao, 3, self.mol.max_memory):
            grid_ao[:, current_grid_count:current_grid_count+ao.shape[1]] = ao
            current_grid_count += ao.shape[1]
        grid_ao_0 = grid_ao[0]
        grid_ao_1 = grid_ao[1:4]
        grid_ao_2T = grid_ao[4:10]
        XX, XY, XZ, YY, YZ, ZZ = range(4, 10)
        XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
        grid_ao_2 = np.array([
            [grid_ao[XX], grid_ao[XY], grid_ao[XZ]],
            [grid_ao[XY], grid_ao[YY], grid_ao[YZ]],
            [grid_ao[XZ], grid_ao[YZ], grid_ao[ZZ]],
        ])
        grid_ao_3T = np.array([
            [grid_ao[XXX], grid_ao[XXY], grid_ao[XXZ], grid_ao[XYY], grid_ao[XYZ], grid_ao[XZZ]],
            [grid_ao[XXY], grid_ao[XYY], grid_ao[XYZ], grid_ao[YYY], grid_ao[YYZ], grid_ao[YZZ]],
            [grid_ao[XXZ], grid_ao[XYZ], grid_ao[XZZ], grid_ao[YYZ], grid_ao[YZZ], grid_ao[ZZZ]],
        ])
        grid_ao_3 = np.array([
            [[grid_ao[XXX], grid_ao[XXY], grid_ao[XXZ]],
             [grid_ao[XXY], grid_ao[XYY], grid_ao[XYZ]],
             [grid_ao[XXZ], grid_ao[XYZ], grid_ao[XZZ]]],
            [[grid_ao[XXY], grid_ao[XYY], grid_ao[XYZ]],
             [grid_ao[XYY], grid_ao[YYY], grid_ao[YYZ]],
             [grid_ao[XYZ], grid_ao[YYZ], grid_ao[YZZ]]],
            [[grid_ao[XXZ], grid_ao[XYZ], grid_ao[XZZ]],
             [grid_ao[XYZ], grid_ao[YYZ], grid_ao[YZZ]],
             [grid_ao[XZZ], grid_ao[YZZ], grid_ao[ZZZ]]],
        ])
        grid_rho_01 = np.einsum("uv, rgu, gv -> rg", D, grid_ao[0:4], grid_ao_0)
        grid_rho_01[1:] *= 2
        grid_rho_0 = grid_rho_01[0]
        grid_rho_1 = grid_rho_01[1:4]
        grid_rho_2 = (
            + 2 * np.einsum("uv, rgu, wgv -> rwg", D, grid_ao_1, grid_ao_1)
            + 2 * np.einsum("uv, rwgu, gv -> rwg", D, grid_ao_2, grid_ao_0)
        )
        grid_rho_3 = (
            + 2 * np.einsum("uv, rwxgu, gv -> rwxg", D, grid_ao_3, grid_ao_0)
            + 2 * np.einsum("uv, rwgu, xgv -> rwxg", D, grid_ao_2, grid_ao_1)
            + 2 * np.einsum("uv, rxgu, wgv -> rwxg", D, grid_ao_2, grid_ao_1)
            + 2 * np.einsum("uv, wxgu, rgv -> rwxg", D, grid_ao_2, grid_ao_1)
        )

        natm = mol.natm
        grid_A_rho_1 = np.zeros((natm, 3, ngrid))
        grid_A_rho_2 = np.zeros((natm, 3, 3, ngrid))
        for A in range(natm):
            _, _, p0, p1 = mol.aoslice_by_atom()[A]
            sA = slice(p0, p1)
            grid_A_rho_1[A] = - 2 * np.einsum("tgk, gl, kl -> tg ", grid_ao_1[:, :, sA], grid_ao_0, D[sA])
            grid_A_rho_2[A] = - 2 * np.einsum("trgk, gl, kl -> trg", grid_ao_2[:, :, :, sA], grid_ao_0, D[sA])
            grid_A_rho_2[A] += - 2 * np.einsum("tgk, rgl, kl -> trg", grid_ao_1[:, :, sA], grid_ao_1, D[sA])

        grid_A_gamma_1 = 2 * np.einsum("rg, Atrg -> Atg", grid_rho_1, grid_A_rho_2)

        grid_AB_rho_2 = np.zeros((natm, natm, 3, 3, ngrid))
        grid_AB_rho_3 = np.zeros((natm, natm, 3, 3, 3, ngrid))
        for A in range(natm):
            _, _, p0A, p1A = mol.aoslice_by_atom()[A]
            sA = slice(p0A, p1A)

            grid_AB_rho_2[A, A] += 2 * np.einsum("tsgu, gv, uv -> tsg", grid_ao_2[:, :, :, sA], grid_ao_0, D[sA])
            grid_AB_rho_3[A, A] += 2 * np.einsum("tsgu, rgv, uv -> tsrg", grid_ao_2[:, :, :, sA], grid_ao_1, D[sA])
            grid_AB_rho_3[A, A] += 2 * np.einsum("tsrgu, gv, uv -> tsrg", grid_ao_3[:, :, :, :, sA], grid_ao_0, D[sA])

            for B in range(A + 1):
                _, _, p0B, p1B = mol.aoslice_by_atom()[B]
                sB = slice(p0B, p1B)

                grid_AB_rho_2[A, B] += 2 * np.einsum("tgu, sgv, uv -> tsg",
                                                     grid_ao_1[:, :, sA], grid_ao_1[:, :, sB], D[sA, sB])
                grid_AB_rho_3[A, B] += 2 * np.einsum("tgu, srgv, uv -> tsrg",
                                                     grid_ao_1[:, :, sA], grid_ao_2[:, :, :, sB], D[sA, sB])
                grid_AB_rho_3[A, B] += 2 * np.einsum("trgu, sgv, uv -> tsrg",
                                                     grid_ao_2[:, :, :, sA], grid_ao_1[:, :, sB], D[sA, sB])
                if A != B:
                    grid_AB_rho_2[B, A] = grid_AB_rho_2[A, B].swapaxes(0, 1)
                    grid_AB_rho_3[B, A] = grid_AB_rho_3[A, B].swapaxes(0, 1)

        grid_AB_gamma_2 = (
            + 2 * np.einsum("Atrg, Bsrg -> ABtsg", grid_A_rho_2, grid_A_rho_2)
            + 2 * np.einsum("rg, ABtsrg -> ABtsg", grid_rho_1, grid_AB_rho_3)
        )

        # Variable definition
        self.ni = ni
        self.ngrid = ngrid
        self.weight = grid_weight
        self.ao = grid_ao
        self.ao_0 = grid_ao_0
        self.ao_1 = grid_ao_1
        self.ao_2T = grid_ao_2T
        self.ao_2 = grid_ao_2
        self.ao_3T = grid_ao_3T
        self.ao_3 = grid_ao_3
        self.rho_01 = grid_rho_01
        self.rho_0 = grid_rho_0
        self.rho_1 = grid_rho_1
        self.rho_2 = grid_rho_2
        self.rho_3 = grid_rho_3
        self.A_rho_1 = grid_A_rho_1
        self.A_rho_2 = grid_A_rho_2
        self.A_gamma_1 = grid_A_gamma_1
        self.AB_rho_2 = grid_AB_rho_2
        self.AB_rho_3 = grid_AB_rho_3
        self.AB_gamma_2 = grid_AB_gamma_2
        return


class GridHelper:

    def __init__(self, mol, grids, D, deriv=3, memory=2000, engine="libxc"):

        self.mol = mol  # type: gto.Mole
        self.grids = grids  # type: dft.Grids
        self.D = D
        self.ni = dft.numint.NumInt()
        if engine == "xcfun":
            from pyscf.dft import xcfun
            self.ni.libxc = xcfun

        self._ao = self.ni.eval_ao(mol, grids.coords, deriv=deriv)
        self._weight = grids.weights
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

    def get_rho_01(self, D=None):
        rho_01 = np.zeros((4, self.ngrid))
        rho_01[0] = self.get_rho_0(D)
        rho_01[1:4] = self.get_rho_1(D)
        return rho_01

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


class KernelHelper:

    def __init__(self, gh, xc, deriv=2):

        # Initialization Parameters
        self.gh = gh  # type: GridHelper or GridIterator or Tuple[GridHelper] or Tuple[GridIterator]
        self.xc = xc  # type: str

        # Variable definition
        self.exc = None
        self.fr = None
        self.fg = None
        self.frr = None
        self.frg = None
        self.fgg = None
        self.frrr = None
        self.frrg = None
        self.frgg = None
        self.fggg = None

        # Calculation
        if type(gh) is not tuple:
            ni = gh.ni
            grid_exc, grid_vxc, grid_fxc, grid_kxc = ni.eval_xc(xc, gh.rho_01, deriv=deriv)
            weight = gh.weight
        else:  # Assume gh is 2-len tuple, for uks calculation
            ni = gh[0].ni
            grid_exc, grid_vxc, grid_fxc, grid_kxc = ni.eval_xc(xc, (gh[0].rho_01, gh[1].rho_01), spin=1, deriv=deriv)
            weight = gh[0].weight
        self.exc = grid_exc * weight
        # transpose here is intended to make uks calculation; however in rks, all transposed vectors are still vectors
        # transpose again, getting vxc, fxc, kxc to use PySCF's _uks_gga_wv*
        # Note! Weight should be set to 1 when calling _uks_gga_wv*
        if deriv >= 1:
            self.fr = grid_vxc[0].T * weight
            self.fg = grid_vxc[1].T * weight
            self.vxc = (self.fr.T, self.fg.T)
        if deriv >= 2:
            self.frr = grid_fxc[0].T * weight
            self.frg = grid_fxc[1].T * weight
            self.fgg = grid_fxc[2].T * weight
            self.fxc = (self.frr.T, self.frg.T, self.fgg.T)
        if deriv >= 3:
            self.frrr = grid_kxc[0].T * weight
            self.frrg = grid_kxc[1].T * weight
            self.frgg = grid_kxc[2].T * weight
            self.fggg = grid_kxc[3].T * weight
            self.kxc = (self.frrr.T, self.frrg.T, self.frgg.T, self.fggg.T)
