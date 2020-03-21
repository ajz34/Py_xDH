from pyscf import dft
import pyscf.dft.numint
from pyscf.dft import xcfun
import numpy as np
from functools import partial
import os
import warnings

from pyxdh.Utilities.grid_iterator import GridIterator

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GridHelper:

    def __init__(self, mol, grids, D):
        warnings.warn("GridHelper is considered memory consuming!")

        # Initialization Parameters
        self.mol = mol  # type: pyscf.gto.Mole
        self.grids = grids  # type: pyscf.dft.gen_grid.Grids
        self.D = D  # type: np.ndarray

        # Calculation
        nao = mol.nao
        ni = dft.numint.NumInt()
        ni.libxc = xcfun
        ngrid = grids.weights.size
        grid_weight = grids.weights
        grid_ao = np.empty((20, ngrid, nao))  # 20 at first dimension is related to 3rd derivative of orbital
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


class KernelHelper:

    def __init__(self, gh, xc, deriv=2):

        # Initialization Parameters
        self.gh = gh  # type: GridHelper or GridIterator
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
        grid_exc, grid_vxc, grid_fxc, grid_kxc = gh.ni.eval_xc(xc, gh.rho_01, deriv=deriv)
        self.exc = grid_exc * gh.weight
        if deriv >= 1:
            self.fr = grid_vxc[0] * gh.weight
            self.fg = grid_vxc[1] * gh.weight
        if deriv >= 2:
            self.frr = grid_fxc[0] * gh.weight
            self.frg = grid_fxc[1] * gh.weight
            self.fgg = grid_fxc[2] * gh.weight
        if deriv >= 3:
            self.frrr = grid_kxc[0] * gh.weight
            self.frrg = grid_kxc[1] * gh.weight
            self.frgg = grid_kxc[2] * gh.weight
            self.fggg = grid_kxc[3] * gh.weight
        return


class Test_GridHelper:

    def test_high_coordinate_derivative_accordance(self):

        from pyscf import gto

        mol = gto.Mole()
        mol.atom = """
        O  0.0  0.0  0.0
        O  0.0  0.0  1.5
        H  1.5  0.0  0.0
        H  0.0  0.7  1.5
        """
        mol.basis = "6-31G"
        mol.verbose = 0
        mol.build()

        grids = dft.gen_grid.Grids(mol)
        grids.atom_grid = (75, 302)
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()

        dmX = np.random.random((mol.nao, mol.nao))
        dmX += dmX.T
        grdh = GridHelper(mol, grids, dmX)
        assert(np.allclose(grdh.A_rho_1.sum(axis=0), - grdh.rho_1))
        assert(np.allclose(grdh.A_rho_2.sum(axis=0), - grdh.rho_2))
        assert(np.allclose(grdh.AB_rho_2.sum(axis=(0, 1)), grdh.rho_2))
        assert(np.allclose(grdh.AB_rho_2, grdh.AB_rho_2.transpose((1, 0, 3, 2, 4))))
        assert(np.allclose(grdh.AB_rho_3.sum(axis=(0, 1)), grdh.rho_3))
        assert(np.allclose(grdh.AB_rho_3, grdh.AB_rho_3.transpose((1, 0, 3, 2, 4, 5))))
        assert(np.allclose(grdh.AB_gamma_2, grdh.AB_gamma_2.transpose((1, 0, 3, 2, 4))))
