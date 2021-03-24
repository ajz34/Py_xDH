from pyxdh.DerivOnce import DerivOnceDFSCF, DerivOnceDFMP2, GradSCF, GradMP2


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




