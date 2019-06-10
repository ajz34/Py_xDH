import pickle

from pyxdh.Utilities.test_molecules import Mol_H2O2
from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator
from pyxdh.DerivOnce import GradMP2


def mol_to_grad_helper(mol):
    print("Processing...")
    H2O2 = Mol_H2O2(mol=mol, xc="0.53*HF + 0.47*B88, 0.73*LYP")
    config = {
        "scf_eng": H2O2.gga_eng,
        "cc": 0.27
    }
    helper = GradMP2(config)
    return helper


if __name__ == '__main__':
    result_dict = {}
    H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
    mol = H2O2.mol

    num_obj = NucCoordDerivGenerator(H2O2.mol, mol_to_grad_helper)
    num_dif = NumericDiff(num_obj, lambda helper: helper.E_1.reshape(-1))
    result_dict["hess"] = num_dif.derivative

    with open("mp2_hessian_b2plyp.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
