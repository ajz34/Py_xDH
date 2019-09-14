import pickle

from pyxdh.Utilities.test_molecules import Mol_H2O2
from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator
from pyxdh.DerivOnce import DipoleNCDFT


def mol_to_grad_helper(mol):
    H2O2 = Mol_H2O2(mol=mol)
    H2O2.hf_eng.kernel()
    config = {
        "scf_eng": H2O2.hf_eng,
        "nc_eng": H2O2.gga_eng
    }
    helper = DipoleNCDFT(config)
    return helper


if __name__ == '__main__':
    result_dict = {}
    H2O2 = Mol_H2O2()
    mol = H2O2.mol

    num_obj = NucCoordDerivGenerator(H2O2.mol, mol_to_grad_helper)
    num_dif = NumericDiff(num_obj, lambda helper: helper.E_1)
    result_dict["dipderiv"] = num_dif.derivative

    with open("ncdft_dipderiv_hf_b3lyp.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
