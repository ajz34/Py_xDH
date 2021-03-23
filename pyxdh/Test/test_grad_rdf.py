import numpy as np
from pyscf import gto, scf, mp, df
from pyxdh.DerivOnce import GradDFSCF, GradDFMP2


class TestGradRDF:

    def test_rdf_rhf_grad(self):
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

    def test_rdf_rhf_eng(self):
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