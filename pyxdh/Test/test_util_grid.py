import numpy as np
from pyscf import gto, dft
from pyxdh.Utilities import GridHelper, GridIterator


class TestGrid:

    def test_high_coordinate_derivative_accordance(self):

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
        assert(np.allclose(grdh.AB_rho_3, grdh.AB_rho_3.transpose((1, 0, 3, 2, 4, 5))))
        assert(np.allclose(grdh.AB_gamma_2, grdh.AB_gamma_2.transpose((1, 0, 3, 2, 4))))

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
