import numpy as np
from functools import partial
import os

from pyscf import gto, scf, mp, df

from pyxdh.DerivOnce import DerivOnceDFSCF, DerivOnceDFMP2, GradSCF, GradMP2

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradDFSCF(DerivOnceDFSCF, GradSCF):

    def _get_E_1(self):
        E_1 = GradSCF._get_E_1(self)
        j_1 = self.scf_grad.get_j(dm=self.D)
        k_1 = self.scf_grad.get_k(dm=self.D)
        v_aux = j_1.aux - 0.5 * self.cx * k_1.aux
        E_1 += v_aux
        return E_1


class GradDFMP2(DerivOnceDFMP2, GradMP2):
    pass


class Test_GradDF:

    def test_DFHF_grad(self):
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
        mf_scf = scf.RHF(mol).density_fit(auxbasis="cc-pVDZ-jkfit").run()
        mf_grad = mf_scf.Gradients().run()

        config = {"scf_eng": mf_scf}
        helper = GradDFSCF(config)
        assert np.allclose(helper.E_1, mf_grad.de, atol=1e-6, rtol=1e-4)

    def test_DFMP2_eng(self):
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
        mf_scf = scf.RHF(mol).density_fit(auxbasis="cc-pVDZ-jkfit").run()
        mf_mp2 = mp.MP2(mf_scf)
        mf_mp2.with_df = df.DF(mol, auxbasis="cc-pVDZ-ri")
        mf_mp2.run()

        aux_ri = df.make_auxmol(mol, "cc-pVDZ-ri")
        config = {"scf_eng": mf_scf, "aux_ri": aux_ri}
        helper = GradDFMP2(config)
        assert np.allclose(helper.eng, mf_mp2.e_tot, rtol=1e-10, atol=1e-12)

