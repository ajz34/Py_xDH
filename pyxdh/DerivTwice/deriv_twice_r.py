# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC, abstractmethod
import warnings
from functools import partial
# pyscf utilities
from pyscf.scf import cphf
# pyxdh utilities
from pyxdh.DerivOnce import DerivOnceSCF, DerivOnceNCDFT, DerivOnceMP2, DerivOnceXDH
from pyxdh.Utilities import cached_property

# additional
einsum = partial(einsum, optimize="greedy")


# Cubic Inheritance: A1
class DerivTwiceSCF(ABC):

    def __init__(self, config):

        # From configuration file, with default values
        self.config = config  # type: dict
        self.A = config["deriv_A"]  # type: DerivOnceSCF
        if self.A_is_B:
            self.B = self.A  # type: DerivOnceSCF
        else:
            self.B = config["deriv_B"]  # type: DerivOnceSCF
        self.rotation = config.get("rotation", True)
        self.grdit_memory = 2000
        if "grdit_memory" in config:
            self.grdit_memory = config["grdit_memory"]

        # Make assertion on coefficient idential of deriv_A and deriv_B instances
        # for some molecules which have degenerate orbital energies,
        # two instances of DerivOnce have different coefficients can be fatal
        assert(np.allclose(self.A.C, self.B.C))
        # After assertion passed, then we can say things may work; however we should not detect intended sabotage
        # So it is recommended to initialize deriv_A and deriv_B with the same runned scf.RHF instance

        # Basic Information

        self.cx = self.A.cx
        self.xc = self.A.xc
        self.grids = self.A.grids
        self.xc_type = self.A.xc_type

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    # region Basic Properties

    @property
    @abstractmethod
    def A_is_B(self) -> bool:
        pass

    @property
    def mol(self):
        return self.A.mol

    @property
    def C(self):
        return self.A.C

    @property
    def Co(self):
        return self.A.Co

    @property
    def Cv(self):
        return self.A.Cv

    @property
    def nmo(self):
        return self.A.nmo

    @property
    def nao(self):
        return self.A.nao

    @property
    def nocc(self):
        return self.A.nocc

    @property
    def nvir(self):
        return self.A.nvir

    @property
    def mo_occ(self):
        return self.A.mo_occ

    @property
    def natm(self):
        return self.A.natm

    @property
    def sa(self):
        return self.A.sa

    @property
    def so(self):
        return self.A.so

    @property
    def sv(self):
        return self.A.sv

    @property
    def e(self):
        return self.A.e

    @property
    def eo(self):
        return self.A.eo

    @property
    def ev(self):
        return self.A.ev

    @property
    def D(self):
        return self.A.D

    # endregion Basic Properties

    # region Properties

    @cached_property
    @abstractmethod
    def H_2_ao(self):
        pass

    @cached_property
    def H_2_mo(self):
        return self.C.T @ self.H_2_ao @ self.C

    @cached_property
    @abstractmethod
    def S_2_ao(self):
        pass

    @cached_property
    def S_2_mo(self):
        return self.C.T @ self.S_2_ao @ self.C

    @cached_property
    @abstractmethod
    def F_2_ao_JKcontrib(self):
        pass

    @cached_property
    def F_2_ao_Jcontrib(self):
        return self.F_2_ao_JKcontrib[0]

    @cached_property
    def F_2_ao_Kcontrib(self):
        return self.F_2_ao_JKcontrib[1]

    @cached_property
    @abstractmethod
    def F_2_ao_GGAcontrib(self):
        pass

    @cached_property
    def F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - 0.5 * self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    @cached_property
    def F_2_mo(self):
        return self.C.T @ self.F_2_ao @ self.C

    @cached_property
    def Xi_2(self):
        A = self.A
        B = self.B
        Xi_2 = (
            self.S_2_mo
            + einsum("Apm, Bqm -> ABpq", A.U_1, B.U_1)
            + einsum("Bpm, Aqm -> ABpq", B.U_1, A.U_1)
            # - einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.S_1_mo)
            # - einsum("Bpm, Aqm -> ABpq", B.S_1_mo, A.S_1_mo)
        )
        if isinstance(A.S_1_mo, np.ndarray) and isinstance(B.S_1_mo, np.ndarray):
            Xi_2 -= einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.S_1_mo)
            Xi_2 -= einsum("Bpm, Aqm -> ABpq", B.S_1_mo, A.S_1_mo)
        return Xi_2

    @cached_property
    def B_2(self):
        A = self.A
        B = self.B
        Ax0_Core = A.Ax0_Core  # Ax0_Core should be the same for A and B derivative

        sa, so = self.sa, self.so
        e = self.e

        B_2 = (
            # line 1
            + self.F_2_mo
            - einsum("ABai, i -> ABai", self.Xi_2, e)
            - 0.5 * Ax0_Core(sa, sa, so, so)(self.Xi_2[:, :, so, so])
            # line 2
            + einsum("Apa, Bpi -> ABai", A.U_1, B.F_1_mo)
            + einsum("Api, Bpa -> ABai", A.U_1, B.F_1_mo)
            + einsum("Bpa, Api -> ABai", B.U_1, A.F_1_mo)
            + einsum("Bpi, Apa -> ABai", B.U_1, A.F_1_mo)
            # line 3
            + einsum("Apa, Bpi, p -> ABai", A.U_1, B.U_1, e)
            + einsum("Bpa, Api, p -> ABai", B.U_1, A.U_1, e)
            # line 4
            + 0.5 * Ax0_Core(sa, sa, sa, sa)(
                + einsum("Akm, Blm -> ABkl", A.U_1[:, :, so], B.U_1[:, :, so])
                + einsum("Bkm, Alm -> ABkl", B.U_1[:, :, so], A.U_1[:, :, so])
            )
            # line 5
            + einsum("Apa, Bpi -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, so]))
            + einsum("Bpa, Api -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, so]))
            # line 6
            + einsum("Api, Bpa -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, so]))
            + einsum("Bpi, Apa -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, so]))
            # line 7
            + A.Ax1_Core(sa, sa, sa, so)(B.U_1[:, :, so])
            + B.Ax1_Core(sa, sa, sa, so)(A.U_1[:, :, so]).swapaxes(0, 1)
        )
        return B_2

    @cached_property
    def U_2(self):
        B_2 = self.B_2
        Xi_2 = self.Xi_2
        Ax0_Core = self.A.Ax0_Core
        sv, so = self.sv, self.so
        nvir, nocc = self.nvir, self.nocc

        # Generate v-o block of U
        U_2_ai = cphf.solve(
            Ax0_Core(sv, so, sv, so, in_cphf=True),
            self.e,
            self.A.scf_eng.mo_occ,
            B_2[:, :, sv, so].reshape(-1, nvir, nocc),
            max_cycle=100,
            tol=self.A.cphf_tol,
            hermi=False
        )[0]
        U_2_ai.shape = (B_2.shape[0], B_2.shape[1], self.nvir, self.nocc)

        # Test whether converged
        conv = (
            + U_2_ai * (self.ev[:, None] - self.eo[None, :])
            + Ax0_Core(sv, so, sv, so)(U_2_ai)
            + self.B_2[:, :, sv, so]
        )
        if abs(conv).max() > 1e-8:
            msg = "\nget_U_2: CP-HF not converged well!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)

        if self.rotation:
            # Generate rotated U
            U_2_pq = - 0.5 * Xi_2
            U_2_pq[:, :, sv, so] = U_2_ai
            U_2_pq[:, :, so, sv] = - Xi_2[:, :, so, sv] - U_2_pq[:, :, sv, so].swapaxes(-1, -2)
        else:
            # Generate total U
            D_pq = - (self.e[:, None] - self.e[None, :]) + 1e-300
            U_2_pq = np.zeros((B_2.shape[0], B_2.shape[1], self.nmo, self.nmo))
            U_2_pq[:, :, sv, so] = U_2_ai
            U_2_pq[:, :, so, sv] = - Xi_2[:, :, so, sv] - U_2_pq[:, :, sv, so].swapaxes(-1, -2)
            U_2_pq[:, :, so, so] = (Ax0_Core(so, so, sv, so)(U_2_ai) + B_2[:, :, so, so]) / D_pq[so, so]
            U_2_pq[:, :, sv, sv] = (Ax0_Core(sv, sv, sv, so)(U_2_ai) + B_2[:, :, sv, sv]) / D_pq[sv, sv]
            for p in range(self.nmo):
                U_2_pq[:, :, p, p] = - Xi_2[:, :, p, p] / 2
            U_2_pq -= (U_2_pq + U_2_pq.swapaxes(-1, -2) + Xi_2) / 2
            U_2_pq -= (U_2_pq + U_2_pq.swapaxes(-1, -2) + Xi_2) / 2

        self._U_2 = U_2_pq
        return self._U_2

    @cached_property
    @abstractmethod
    def eri2_ao(self):
        pass

    @cached_property
    def eri2_mo(self):
        return einsum("ABuvkl, up, vq, kr, ls -> ABpqrs", self.eri2_ao, self.C, self.C, self.C, self.C)

    @cached_property
    def E_2_Skeleton(self):
        return self._get_E_2_Skeleton()

    @cached_property
    def E_2_U(self):
        return self._get_E_2_U()

    @cached_property
    def E_2(self):
        return self._get_E_2()

    @cached_property
    def pdB_F_A_mo(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        pdB_F_A_mo = (
            + self.F_2_mo
            + einsum("Apm, Bmq -> ABpq", A.F_1_mo, B.U_1)
            + einsum("Amq, Bmp -> ABpq", A.F_1_mo, B.U_1)
            + A.Ax1_Core(sa, sa, sa, so)(B.U_1[:, :, so])
        )
        return pdB_F_A_mo

    @cached_property
    def pdB_S_A_mo(self):
        A, B = self.A, self.B
        pdB_S_A_mo = (
            + self.S_2_mo
            + einsum("Apm, Bmq -> ABpq", A.S_1_mo, B.U_1)
            + einsum("Amq, Bmp -> ABpq", A.S_1_mo, B.U_1)
        )
        return pdB_S_A_mo

    @cached_property
    def pdB_B_A(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        Ax0_Core = A.Ax0_Core
        pdB_B_A = (
            + self.pdB_F_A_mo
            - self.pdB_S_A_mo * self.e
            - einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.pdA_F_0_mo)
            - 0.5 * B.Ax1_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]).swapaxes(0, 1)
            - 0.5 * Ax0_Core(sa, sa, so, so)(self.pdB_S_A_mo[:, :, so, so])
            - Ax0_Core(sa, sa, sa, so)(einsum("Bml, Akl -> ABmk", B.U_1[:, :, so], A.S_1_mo[:, so, so]))
            - 0.5 * einsum("Bmp, Amq -> ABpq", B.U_1, Ax0_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]))
            - 0.5 * einsum("Bmq, Amp -> ABpq", B.U_1, Ax0_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]))
        )
        return pdB_B_A

    @cached_property
    def RHS_B(self):
        return self._get_RHS_B()

    def _get_RHS_B(self):
        raise NotImplementedError("In SCF there should be no need to use RHS_B!")

    # endregion

    # region Getters

    @abstractmethod
    def _get_E_2_Skeleton(self):
        pass

    def _get_E_2_U(self):
        A, B = self.A, self.B
        Xi_2 = self.Xi_2
        so, sa = self.so, self.sa
        e, eo = self.e, self.eo
        Ax0_Core = self.A.Ax0_Core

        E_2_U = - 2 * einsum("ABi, i -> AB", Xi_2.diagonal(0, -1, -2)[:, :, so], eo)
        E_2_U += 4 * einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.F_1_mo[:, :, so])
        E_2_U += 4 * einsum("Api, Bpi -> AB", A.U_1[:, :, so], B.F_1_mo[:, :, so])
        E_2_U += 4 * einsum("Api, Bpi, p -> AB", A.U_1[:, :, so], B.U_1[:, :, so], e)
        E_2_U += 4 * einsum("Api, Bpi -> AB", A.U_1[:, :, so], Ax0_Core(sa, so, sa, so)(B.U_1[:, :, so]))

        return E_2_U

    @abstractmethod
    def _get_E_2(self):
        pass

    # endregion


class DerivTwiceNCDFT(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceNCDFT, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceNCDFT classes
        self.A = self.A  # type: DerivOnceNCDFT
        self.B = self.B  # type: DerivOnceNCDFT
        assert(isinstance(self.A, DerivOnceNCDFT))
        assert(isinstance(self.B, DerivOnceNCDFT))

    @cached_property
    def Z(self):
        return self.A.Z

    def _get_RHS_B(self):
        B = self.B
        U_1, Z = B.U_1, B.Z
        so, sv, sa = self.so, self.sv, self.sa
        RHS_B = B.pdA_nc_F_0_mo[:, sv, so].copy()
        RHS_B += B.Ax1_Core(sv, so, sv, so)(Z)
        RHS_B += einsum("Apa, pi -> Aai", U_1[:, :, sv], B.Ax0_Core(sa, so, sv, so)(Z))
        RHS_B += einsum("Api, pa -> Aai", U_1[:, :, so], B.Ax0_Core(sa, sv, sv, so)(Z))
        RHS_B += B.Ax0_Core(sv, so, sa, so)(einsum("Apa, ai -> Api", U_1[:, :, sv], Z))
        RHS_B += B.Ax0_Core(sv, so, sa, sv)(einsum("Api, ai -> Apa", U_1[:, :, so], Z))
        RHS_B += einsum("ci, Aca -> Aai", Z, B.pdA_F_0_mo[:, sv, sv])
        RHS_B -= einsum("ak, Aki -> Aai", Z, B.pdA_F_0_mo[:, so, so])
        return RHS_B

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        E_2_U = 4 * einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * einsum("ABai, ai -> AB", self.pdB_B_A[:, :, sv, so], self.Z)
        E_2_U -= 2 * einsum("Aki, Bki -> AB", A.S_1_mo[:, so, so], B.pdA_nc_F_0_mo[:, so, so])
        E_2_U -= 2 * einsum("ABki, ki -> AB", self.pdB_S_A_mo[:, :, so, so], A.nc_deriv.F_0_mo[so, so])
        return E_2_U


# Cubic Inheritance: C1
class DerivTwiceMP2(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceMP2, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceMP2 classes
        self.A = self.A  # type: DerivOnceMP2
        self.B = self.B  # type: DerivOnceMP2
        assert(isinstance(self.A, DerivOnceMP2))
        assert(isinstance(self.B, DerivOnceMP2))
        assert(self.A.cc == self.B.cc)
        self.cc = self.A.cc

        # For simplicity, these values are not set to be properties
        # However, these values should not be changed or redefined
        self.t_iajb = self.A.t_iajb
        self.T_iajb = self.A.T_iajb
        self.L = self.A.L
        self.D_r = self.A.D_r
        self.W_I = self.A.W_I
        self.D_iajb = self.A.D_iajb

    # region Properties

    @cached_property
    def pdB_pdpA_eri0_iajb(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        eri1_mo, U_1 = A.eri1_mo, B.U_1
        pdB_pdpA_eri0_iajb = (
            + einsum("ABuvkl, up, vq, kr, ls -> ABpqrs", self.eri2_ao, self.Co, self.Cv, self.Co, self.Cv)
            + einsum("Apjkl, Bpi -> ABijkl", eri1_mo[:, :, sv, so, sv], U_1[:, :, so])
            + einsum("Aipkl, Bpj -> ABijkl", eri1_mo[:, so, :, so, sv], U_1[:, :, sv])
            + einsum("Aijpl, Bpk -> ABijkl", eri1_mo[:, so, sv, :, sv], U_1[:, :, so])
            + einsum("Aijkp, Bpl -> ABijkl", eri1_mo[:, so, sv, so, :], U_1[:, :, sv])
        )
        return pdB_pdpA_eri0_iajb

    def _get_RHS_B(self):
        B = self.B
        so, sv, sa = self.so, self.sv, self.sa
        U_1, D_r, pdB_F_0_mo, eri0_mo = B.U_1, B.D_r, B.pdA_F_0_mo, B.eri0_mo
        Ax0_Core, Ax1_Core = B.Ax0_Core, B.Ax1_Core
        pdB_D_r_oovv = B.pdA_D_r_oovv

        # total partial eri0
        pdA_eri0_mo = B.pdA_eri0_mo

        RHS_B = np.zeros((U_1.shape[0], self.nvir, self.nocc))
        # D_r Part
        RHS_B += Ax0_Core(sv, so, sa, sa)(pdB_D_r_oovv)
        RHS_B += Ax1_Core(sv, so, sa, sa)(D_r)
        RHS_B += einsum("Apa, pi -> Aai", U_1[:, :, sv], Ax0_Core(sa, so, sa, sa)(D_r))
        RHS_B += einsum("Api, ap -> Aai", U_1[:, :, so], Ax0_Core(sv, sa, sa, sa)(D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(einsum("Amp, pq -> Amq", U_1, D_r))
        RHS_B += Ax0_Core(sv, so, sa, sa)(einsum("Amq, pq -> Apm", U_1, D_r))
        # (ea - ei) * Dai
        RHS_B += einsum("Aca, ci -> Aai", pdB_F_0_mo[:, sv, sv], D_r[sv, so])
        RHS_B -= einsum("Aki, ak -> Aai", pdB_F_0_mo[:, so, so], D_r[sv, so])
        # 2-pdm part
        RHS_B -= 4 * einsum("Ajakb, ijbk -> Aai", B.pdA_T_iajb, eri0_mo[so, so, sv, so])
        RHS_B += 4 * einsum("Aibjc, abjc -> Aai", B.pdA_T_iajb, eri0_mo[sv, sv, so, sv])
        RHS_B -= 4 * einsum("jakb, Aijbk -> Aai", B.T_iajb, pdA_eri0_mo[:, so, so, sv, so])
        RHS_B += 4 * einsum("ibjc, Aabjc -> Aai", B.T_iajb, pdA_eri0_mo[:, sv, sv, so, sv])

        return RHS_B

    def _get_E_2_MP2_Contrib(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv

        E_2_MP2_Contrib = (
            # D_r * B
            + einsum("pq, ABpq -> AB", self.D_r, self.pdB_B_A)
            + einsum("Bpq, Apq -> AB", B.pdA_D_r_oovv, A.B_1)
            + einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
            # W_I * S
            + einsum("pq, ABpq -> AB", self.W_I, self.pdB_S_A_mo)
            + einsum("Bpq, Apq -> AB", B.pdA_W_I, A.S_1_mo)
            # T * g
            + 2 * einsum("Biajb, Aiajb -> AB", B.pdA_T_iajb, A.eri1_mo[:, so, sv, so, sv])
            + 2 * einsum("iajb, ABiajb -> AB", self.T_iajb, self.pdB_pdpA_eri0_iajb)
        )
        return E_2_MP2_Contrib

    def _get_E_2(self):
        return super(DerivTwiceMP2, self)._get_E_2() + self._get_E_2_MP2_Contrib()

    # endregion


class DerivTwiceXDH(DerivTwiceMP2, DerivTwiceNCDFT, ABC):

    def __init__(self, config):
        super(DerivTwiceXDH, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceXDH classes
        self.A = self.A  # type: DerivOnceXDH
        self.B = self.B  # type: DerivOnceXDH
        assert(isinstance(self.A, DerivOnceXDH))
        assert(isinstance(self.B, DerivOnceXDH))

    def _get_RHS_B(self):
        RHS_B = DerivTwiceMP2._get_RHS_B(self)
        RHS_B += 4 * self.B.pdA_nc_F_0_mo[:, self.sv, self.so]
        return RHS_B

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so = self.so
        E_2_U = 4 * einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U -= 2 * einsum("Aki, Bki -> AB", A.S_1_mo[:, so, so], B.pdA_nc_F_0_mo[:, so, so])
        E_2_U -= 2 * einsum("ABki, ki -> AB", self.pdB_S_A_mo[:, :, so, so], A.nc_deriv.F_0_mo[so, so])
        return E_2_U
