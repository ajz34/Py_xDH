import pickle
from pyscf import scf, gto, dft
from pyxdh.DerivOnce import DipoleNCDFT
from pyxdh.Utilities import NumericDiff, DipoleDerivGenerator


def dipole_generator(component, interval):
    mol = gto.Mole(atom="N 0. 0. 0.; H 1. 0. 0.; H 0. 2. 0.; H 0. 0. 1.5", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.build()
    scf_eng = scf.RHF(mol)
    nc_eng = dft.RKS(mol, xc="B3LYPg")
    nc_eng.grids = grids

    def get_hcore(mol=mol):
        return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]

    scf_eng.get_hcore = get_hcore
    nc_eng.get_hcore = get_hcore
    scf_eng.run()
    diph = DipoleNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng})
    return diph.E_1


if __name__ == '__main__':
    name = __file__.split("/")[-1].split(".")[0]
    mol = gto.Mole(atom="N 0. 0. 0.; H 1.5 0. 0.2; H 0.1 1.2 0.; H 0. 0. 1.", basis="6-31G", verbose=0).build()
    num_obj = DipoleDerivGenerator(dipole_generator)
    num_dif = NumericDiff(num_obj).derivative
    with open(name + ".dat", "wb") as f:
        pickle.dump(- num_dif, f, pickle.HIGHEST_PROTOCOL)
