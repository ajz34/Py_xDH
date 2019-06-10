from pyxdh.Utilities import DipoleDerivGenerator, NumericDiff
from pyxdh.Utilities.test_molecules import Mol_H2O2
from pyxdh.DerivOnce import DipoleXDH
from pyscf import scf
import pickle


def dipole_generator(component, interval):
    H2O2_sc = Mol_H2O2(xc="B3LYPg")
    H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
    mol = H2O2_sc.mol

    def get_hcore(mol=mol):
        return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]

    H2O2_sc.gga_eng.get_hcore = get_hcore
    H2O2_nc.gga_eng.get_hcore = get_hcore

    config = {
        "scf_eng": H2O2_sc.gga_eng,
        "nc_eng": H2O2_nc.gga_eng,
        "cc": 0.3211,
    }
    helper = DipoleXDH(config)
    return helper


if __name__ == '__main__':

    num_obj = DipoleDerivGenerator(dipole_generator)
    num_dif = NumericDiff(num_obj, lambda helper: helper.E_1)

    result_dict = {"polarizability": - num_dif.derivative}

    with open("xdh_polarizability_xyg3.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
