import pickle
from pyxdh.Utilities.test_molecules import Mol_H2O2
from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator
from pyxdh.DerivOnce import DipoleXDH


def mol_to_dip_helper(mol):
    print("Processing...")
    H2O2_sc = Mol_H2O2(mol=mol, xc="B3LYPg")
    H2O2_nc = Mol_H2O2(mol=mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
    config = {
        "scf_eng": H2O2_sc.gga_eng,
        "nc_eng": H2O2_nc.gga_eng,
        "cc": 0.3211
    }
    helper = DipoleXDH(config)
    return helper


if __name__ == '__main__':
    result_dict = {}
    H2O2 = Mol_H2O2()
    mol = H2O2.mol

    num_obj = NucCoordDerivGenerator(H2O2.mol, mol_to_dip_helper)
    num_dif = NumericDiff(num_obj, lambda helper: helper.E_1)
    result_dict["dipderiv"] = num_dif.derivative

    with open("xdh_dipderiv_xyg3.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
