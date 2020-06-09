import pickle
from pyscf import scf
import numpy as np

from pyxdh.Utilities.test_molecules import Mol_CH3
from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator, DipoleDerivGenerator
from pyxdh.DerivOnce import GradUNCDFT


def mol_to_grad_helper(mol):
    CH3 = Mol_CH3(mol=mol)
    CH3.hf_eng.kernel()
    config = {
        "scf_eng": CH3.hf_eng,
        "nc_eng": CH3.gga_eng
    }
    helper = GradUNCDFT(config)
    return helper


def dipole_generator(component, interval):
    CH3 = Mol_CH3()
    mol = CH3.mol

    def get_hcore(mol=mol):
        return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]

    CH3.hf_eng.get_hcore = get_hcore
    CH3.gga_eng.get_hcore = get_hcore

    config = {
        "scf_eng": CH3.hf_eng,
        "nc_eng": CH3.gga_eng
    }
    helper = GradUNCDFT(config)
    return helper


if __name__ == '__main__':
    result_dict = {}
    CH3 = Mol_CH3()
    mol = CH3.mol

    num_obj = NucCoordDerivGenerator(CH3.mol, mol_to_grad_helper)
    num_dif = NumericDiff(num_obj, lambda helper: helper.eng)
    result_dict["grad"] = num_dif.derivative

    num_obj = DipoleDerivGenerator(dipole_generator)
    num_dif = NumericDiff(num_obj, lambda helper: helper.eng)
    dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
    result_dict["dipole"] = num_dif.derivative + dip_nuc

    with open("ncdft_derivonce_uhf_ub3lyp.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
