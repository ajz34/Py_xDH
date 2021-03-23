import pickle
from pyscf import scf, gto, dft
from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator


def mol_to_eng(mol):
    grids = dft.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.build()
    scf_eng = scf.RHF(mol).run()
    nc_eng = dft.RKS(mol, xc="B3LYPg")
    nc_eng.grids = grids
    return nc_eng.energy_tot(dm=scf_eng.make_rdm1())


if __name__ == '__main__':
    name = __file__.split("/")[-1].split(".")[0]
    mol = gto.Mole(atom="N 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", verbose=0).build()
    num_obj = NucCoordDerivGenerator(mol, mol_to_eng)
    num_dif = NumericDiff(num_obj).derivative
    with open(name + ".dat", "wb") as f:
        pickle.dump(num_dif, f, pickle.HIGHEST_PROTOCOL)
