from pyxdh.Utilities import DipoleDerivGenerator, NumericDiff
from pyxdh.Utilities.test_molecules import Mol_H2O2
from pyxdh.DerivOnce import DipoleNCDFT
from pyscf import scf
import pickle


def dipole_generator(component, interval):
    H2O2 = Mol_H2O2()
    mol = H2O2.mol

    def get_hcore(mol=mol):
        return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]

    H2O2.hf_eng.get_hcore = get_hcore
    H2O2.gga_eng.get_hcore = get_hcore

    config = {
        "scf_eng": H2O2.hf_eng,
        "nc_eng": H2O2.gga_eng,
    }
    helper = DipoleNCDFT(config)
    return helper


if __name__ == '__main__':

    num_obj = DipoleDerivGenerator(dipole_generator)
    num_dif = NumericDiff(num_obj, lambda helper: helper.E_1)

    result_dict = {"polarizability": - num_dif.derivative}

    with open("ncdft_polarizability_hf_b3lyp.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
