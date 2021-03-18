import pickle
from pyscf import gto, scf, df, mp
import numpy as np

from pyxdh.Utilities import NumericDiff, NucCoordDerivGenerator


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = """
    N  0.  0.  0.
    H  1.5 0.  0.2
    H  0.1 1.2 0.
    H  0.  0.  1.
    """
    mol.basis = "cc-pVDZ"
    mol.verbose = 0
    mol.build()

    def mol_to_eng_true(mol):
        mf_scf = scf.RHF(mol).density_fit(auxbasis="cc-pVDZ-jkfit").run()
        mf_mp2 = mp.MP2(mf_scf)
        mf_mp2.with_df = df.DF(mol, auxbasis="cc-pVDZ-ri")
        return mf_mp2.run().e_tot

    def mol_to_eng_false(mol):
        mf_scf = scf.RHF(mol).density_fit(auxbasis="cc-pVDZ-jkfit").run()
        mf_mp2 = mp.MP2(mf_scf)
        return mf_mp2.run().e_tot

    # Check deviation of using jkfit and ri auxbasis in MP2 gradient
    num_true = NucCoordDerivGenerator(mol, mol_to_eng_true)
    num_false = NucCoordDerivGenerator(mol, mol_to_eng_false)
    val_true = NumericDiff(num_true).derivative
    val_false = NumericDiff(num_false).derivative
    assert(np.allclose(val_true, val_false, atol=1e-5, rtol=1e-4) is False)

    result_dict = {"grad": val_true}
    with open("df_mp2_grad_mp2.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)

