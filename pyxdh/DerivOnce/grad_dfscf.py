import numpy as np
from functools import partial
import os

from pyscf import gto, scf

from pyxdh.DerivOnce import DerivOnceDFSCF, GradSCF

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradDFSCF(DerivOnceDFSCF, GradSCF):
    pass


class Test_GradDFSCF:

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
        np.allclose(helper.E_1, mf_grad.de, atol=1e-6, rtol=1e-4)




