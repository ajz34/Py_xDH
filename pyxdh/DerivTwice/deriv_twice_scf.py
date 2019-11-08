import numpy as np
from pyscf.scf import cphf
from abc import ABC, abstractmethod
from functools import partial
import warnings
import os

from pyxdh.DerivOnce import DerivOnceSCF, DerivOnceNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: A1
class DerivTwiceSCF(ABC):

    def __init__(self, config):

        # From configuration file, with default values
        self.config = config  # type: dict
        self.A = config["deriv_A"]  # type: DerivOnceSCF
        self.B = config["deriv_B"]  # type: DerivOnceSCF
        self.rotation = config.get("rotation", True)
        self.grdit_memory = 2000
        if "grdit_memory" in config:
            self.grdit_memory = config["grdit_memory"]

        self._same_deriv = True  # This is only an indication to use symmetry to eliminate computation cost

        # Make assertion on coefficient idential of deriv_A and deriv_B instances
        # for some molecules which have degenerate orbital energies,
        # two instances of DerivOnce have different coefficients can be fatal
        assert(np.allclose(self.A.C, self.B.C))
        # After assertion passed, then we can say things may work; however we should not detect intended sabotage
        # So it is recommended to initialize deriv_A and deriv_B with the same runned scf.RHF instance

        # Basic Information
        self._mol = self.A.mol
        self._C = self.A.C
        self._e = self.A.e
        self._D = self.A.D
        self._mo_occ = self.A.mo_occ

        self.cx = self.A.cx
        self.xc = self.A.xc
        self.grids = self.A.grids
        self.xc_type = self.A.xc_type

        # Matrices
        self._H_2_ao = NotImplemented
        self._H_2_mo = NotImplemented
        self._S_2_ao = NotImplemented
        self._S_2_mo = NotImplemented
        self._F_2_ao = NotImplemented
        self._F_2_ao_Jcontrib = NotImplemented
        self._F_2_ao_Kcontrib = NotImplemented
        self._F_2_ao_GGAcontrib = NotImplemented
        self._F_2_mo = NotImplemented
        self._Xi_2 = NotImplemented
        self._B_2 = NotImplemented
        self._U_2 = NotImplemented
        self._eri2_ao = NotImplemented
        self._eri2_mo = NotImplemented

        # E_2
        self._E_2_Skeleton = NotImplemented
        self._E_2_U = NotImplemented
        self._E_2 = NotImplemented

        # Intermediate variables
        self._pdB_F_A_mo = NotImplemented
        self._pdB_S_A_mo = NotImplemented
        self._pdB_B_A = NotImplemented

    # region Properties

    @property
    def mol(self):
        return self._mol

    @property
    def C(self):
        return self._C

    @property
    def Co(self):
        return self.C[:, self.so]

    @property
    def Cv(self):
        return self.C[:, self.sv]

    @property
    def nmo(self):
        return self.C.shape[1]

    @property
    def nao(self):
        return self.A.nao

    @property
    def nocc(self):
        return self.mol.nelec[0]

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def mo_occ(self):
        return self._mo_occ

    @property
    def natm(self):
        return self.mol.natm

    @property
    def sa(self):
        return slice(0, self.nmo)

    @property
    def so(self):
        return slice(0, self.nocc)

    @property
    def sv(self):
        return slice(self.nocc, self.nmo)

    @property
    def e(self):
        return self._e

    @property
    def eo(self):
        return self.e[self.so]

    @property
    def ev(self):
        return self.e[self.sv]

    @property
    def D(self):
        return self._D

    @property
    def H_2_ao(self):
        if self._H_2_ao is NotImplemented:
            self._H_2_ao = self._get_H_2_ao()
        return self._H_2_ao

    @property
    def H_2_mo(self):
        if self._H_2_mo is NotImplemented:
            self._H_2_mo = self._get_H_2_mo()
        return self._H_2_mo

    @property
    def S_2_ao(self):
        if self._S_2_ao is NotImplemented:
            self._S_2_ao = self._get_S_2_ao()
        return self._S_2_ao

    @property
    def S_2_mo(self):
        if self._S_2_mo is NotImplemented:
            self._S_2_mo = self._get_S_2_mo()
        return self._S_2_mo

    @property
    def F_2_ao_Jcontrib(self):
        if self._F_2_ao_Jcontrib is NotImplemented:
            self._F_2_ao_Jcontrib, self._F_2_ao_Kcontrib = self._get_F_2_ao_JKcontrib()
        return self._F_2_ao_Jcontrib

    @property
    def F_2_ao_Kcontrib(self):
        if self._F_2_ao_Kcontrib is NotImplemented:
            self._F_2_ao_Jcontrib, self._F_2_ao_Kcontrib = self._get_F_2_ao_JKcontrib()
        return self._F_2_ao_Kcontrib

    @property
    def F_2_ao_GGAcontrib(self):
        if self._F_2_ao_GGAcontrib is NotImplemented:
            self._F_2_ao_GGAcontrib = self._get_F_2_ao_GGAcontrib()
        return self._F_2_ao_GGAcontrib

    @property
    def F_2_ao(self):
        if self._F_2_ao is NotImplemented:
            self._F_2_ao = self._get_F_2_ao()
        return self._F_2_ao

    @property
    def F_2_mo(self):
        if self._F_2_mo is NotImplemented:
            self._F_2_mo = self._get_F_2_mo()
        return self._F_2_mo

    @property
    def Xi_2(self):
        if self._Xi_2 is NotImplemented:
            self._Xi_2 = self._get_Xi_2()
        return self._Xi_2

    @property
    def B_2(self):
        if self._B_2 is NotImplemented:
            self._B_2 = self._get_B_2()
        return self._B_2

    @property
    def U_2(self):
        if self._U_2 is NotImplemented:
            self._U_2 = self._get_U_2()
        return self._U_2

    @property
    def eri2_ao(self):
        if self._eri2_ao is NotImplemented:
            self._eri2_ao = self._get_eri2_ao()
        return self._eri2_ao

    @property
    def eri2_mo(self):
        if self._eri2_mo is NotImplemented:
            self._eri2_mo = self._get_eri2_mo()
        return self._eri2_mo

    @property
    def E_2_Skeleton(self):
        if self._E_2_Skeleton is NotImplemented:
            self._E_2_Skeleton = self._get_E_2_Skeleton()
        return self._E_2_Skeleton

    @property
    def E_2_U(self):
        if self._E_2_U is NotImplemented:
            self._E_2_U = self._get_E_2_U()
        return self._E_2_U

    @property
    def E_2(self):
        if self._E_2 is NotImplemented:
            self._E_2 = self._get_E_2()
        return self._E_2

    @property
    def pdB_F_A_mo(self):
        if self._pdB_F_A_mo is NotImplemented:
            self._pdB_F_A_mo = self._get_pdB_F_A_mo()
        return self._pdB_F_A_mo

    @property
    def pdB_S_A_mo(self):
        if self._pdB_S_A_mo is NotImplemented:
            self._pdB_S_A_mo = self._get_pdB_S_A_mo()
        return self._pdB_S_A_mo

    @property
    def pdB_B_A(self):
        if self._pdB_B_A is NotImplemented:
            self._pdB_B_A = self._get_pdB_B_A()
        return self._pdB_B_A

    @property
    def RHS_B(self):
        if self._RHS_B is NotImplemented:
            self._RHS_B = self._get_RHS_B()
        return self._RHS_B

    # endregion

    # region Getters

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    @abstractmethod
    def _get_H_2_ao(self):
        pass

    def _get_H_2_mo(self):
        return self.C.T @ self.H_2_ao @ self.C

    @abstractmethod
    def _get_S_2_ao(self):
        pass

    def _get_S_2_mo(self):
        if not isinstance(self.S_2_ao, np.ndarray):
            return 0
        return self.C.T @ self.S_2_ao @ self.C

    @abstractmethod
    def _get_F_2_ao_JKcontrib(self):
        pass

    @abstractmethod
    def _get_F_2_ao_GGAcontrib(self):
        pass

    def _get_F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - 0.5 * self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    def _get_F_2_mo(self):
        return self.C.T @ self.F_2_ao @ self.C

    def _get_Xi_2(self):
        A = self.A
        B = self.B

        Xi_2 = (
            self.S_2_mo
            + np.einsum("Apm, Bqm -> ABpq", A.U_1, B.U_1)
            + np.einsum("Bpm, Aqm -> ABpq", B.U_1, A.U_1)
            # - np.einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.S_1_mo)
            # - np.einsum("Bpm, Aqm -> ABpq", B.S_1_mo, A.S_1_mo)
        )
        if isinstance(A.S_1_mo, np.ndarray) and isinstance(B.S_1_mo, np.ndarray):
            Xi_2 -= np.einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.S_1_mo)
            Xi_2 -= np.einsum("Bpm, Aqm -> ABpq", B.S_1_mo, A.S_1_mo)
        return Xi_2

    def _get_B_2(self):
        A = self.A
        B = self.B
        Ax0_Core = A.Ax0_Core  # Ax0_Core should be the same for A and B derivative

        sa, so = self.sa, self.so
        e = self.e

        B_2 = (
            # line 1
            + self.F_2_mo
            - np.einsum("ABai, i -> ABai", self.Xi_2, e)
            - 0.5 * Ax0_Core(sa, sa, so, so)(self.Xi_2[:, :, so, so])
            # line 2
            + np.einsum("Apa, Bpi -> ABai", A.U_1, B.F_1_mo)
            + np.einsum("Api, Bpa -> ABai", A.U_1, B.F_1_mo)
            + np.einsum("Bpa, Api -> ABai", B.U_1, A.F_1_mo)
            + np.einsum("Bpi, Apa -> ABai", B.U_1, A.F_1_mo)
            # line 3
            + np.einsum("Apa, Bpi, p -> ABai", A.U_1, B.U_1, e)
            + np.einsum("Bpa, Api, p -> ABai", B.U_1, A.U_1, e)
            # line 4
            + 0.5 * Ax0_Core(sa, sa, sa, sa)(
                + np.einsum("Akm, Blm -> ABkl", A.U_1[:, :, so], B.U_1[:, :, so])
                + np.einsum("Bkm, Alm -> ABkl", B.U_1[:, :, so], A.U_1[:, :, so])
            )
            # line 5
            + np.einsum("Apa, Bpi -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, so]))
            + np.einsum("Bpa, Api -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, so]))
            # line 6
            + np.einsum("Api, Bpa -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, so]))
            + np.einsum("Bpi, Apa -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, so]))
        )
        if self.xc_type != "HF":
            B_2 += (
                # line 7
                + A.Ax1_Core(sa, sa, sa, so)(B.U_1[:, :, so])
                + B.Ax1_Core(sa, sa, sa, so)(A.U_1[:, :, so]).swapaxes(0, 1)
            )
        return B_2

    def _get_U_2(self):
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

    @abstractmethod
    def _get_eri2_ao(self):
        pass

    def _get_eri2_mo(self):
        return np.einsum("ABuvkl, up, vq, kr, ls -> ABpqrs", self.eri2_ao, self.C, self.C, self.C, self.C)

    def _get_pdB_F_A_mo(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        pdB_F_A_mo = (
            + self.F_2_mo
            + np.einsum("Apm, Bmq -> ABpq", A.F_1_mo, B.U_1)
            + np.einsum("Amq, Bmp -> ABpq", A.F_1_mo, B.U_1)
            + A.Ax1_Core(sa, sa, sa, so)(B.U_1[:, :, so])
        )
        return pdB_F_A_mo

    def _get_pdB_S_A_mo(self):
        A, B = self.A, self.B
        pdB_S_A_mo = (
            + self.S_2_mo
            + np.einsum("Apm, Bmq -> ABpq", A.S_1_mo, B.U_1)
            + np.einsum("Amq, Bmp -> ABpq", A.S_1_mo, B.U_1)
        )
        return pdB_S_A_mo

    def _get_pdB_B_A(self):
        A, B = self.A, self.B
        so, sa = self.so, self.sa
        Ax0_Core = A.Ax0_Core
        pdB_B_A = (
            + self.pdB_F_A_mo
            - self.pdB_S_A_mo * self.e
            - np.einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.pdA_F_0_mo)
            - 0.5 * B.Ax1_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]).swapaxes(0, 1)
            - 0.5 * Ax0_Core(sa, sa, so, so)(self.pdB_S_A_mo[:, :, so, so])
            - Ax0_Core(sa, sa, sa, so)(np.einsum("Bml, Akl -> ABmk", B.U_1[:, :, so], A.S_1_mo[:, so, so]))
            - 0.5 * np.einsum("Bmp, Amq -> ABpq", B.U_1, Ax0_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]))
            - 0.5 * np.einsum("Bmq, Amp -> ABpq", B.U_1, Ax0_Core(sa, sa, so, so)(A.S_1_mo[:, so, so]))
        )
        return pdB_B_A

    def _get_RHS_B(self):
        return "In SCF there should be no need to use RHS_B!"

    @abstractmethod
    def _get_E_2_Skeleton(self):
        pass

    def _get_E_2_U(self):
        A, B = self.A, self.B
        Xi_2 = self.Xi_2
        so, sa = self.so, self.sa
        e, eo = self.e, self.eo
        Ax0_Core = self.A.Ax0_Core

        E_2_U = - 2 * np.einsum("ABi, i -> AB", Xi_2.diagonal(0, -1, -2)[:, :, so], eo)
        E_2_U += 4 * np.einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.F_1_mo[:, :, so])
        E_2_U += 4 * np.einsum("Api, Bpi -> AB", A.U_1[:, :, so], B.F_1_mo[:, :, so])
        E_2_U += 4 * np.einsum("Api, Bpi, p -> AB", A.U_1[:, :, so], B.U_1[:, :, so], e)
        E_2_U += 4 * np.einsum("Api, Bpi -> AB", A.U_1[:, :, so], Ax0_Core(sa, so, sa, so)(B.U_1[:, :, so]))

        return E_2_U

    @abstractmethod
    def _get_E_2(self):
        pass

    # endregion


class DerivTwiceNCDFT(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceNCDFT, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceNCDFT classes
        self.A = config["deriv_A"]  # type: DerivOnceNCDFT
        self.B = config["deriv_B"]  # type: DerivOnceNCDFT
        assert(isinstance(self.A, DerivOnceNCDFT))
        assert(isinstance(self.B, DerivOnceNCDFT))

        # For simplicity, these values are not set to be properties
        # However, these values should not be changed or redefined
        self._Z = NotImplemented

        # For non-consistent calculation
        self._RHS_B = NotImplemented

    @property
    def Z(self):
        if self._Z is NotImplemented:
            self._Z = self.A.Z
        return self._Z

    def _get_RHS_B(self):
        B = self.B
        U_1, Z = B.U_1, B.Z
        so, sv, sa = self.so, self.sv, self.sa
        RHS_B = B.pdA_nc_F_0_mo[:, sv, so].copy()
        RHS_B += B.Ax1_Core(sv, so, sv, so)(Z)
        RHS_B += np.einsum("Apa, pi -> Aai", U_1[:, :, sv], B.Ax0_Core(sa, so, sv, so)(Z))
        RHS_B += np.einsum("Api, pa -> Aai", U_1[:, :, so], B.Ax0_Core(sa, sv, sv, so)(Z))
        RHS_B += B.Ax0_Core(sv, so, sa, so)(np.einsum("Apa, ai -> Api", U_1[:, :, sv], Z))
        RHS_B += B.Ax0_Core(sv, so, sa, sv)(np.einsum("Api, ai -> Apa", U_1[:, :, so], Z))
        RHS_B += np.einsum("ci, Aca -> Aai", Z, B.pdA_F_0_mo[:, sv, sv])
        RHS_B -= np.einsum("ak, Aki -> Aai", Z, B.pdA_F_0_mo[:, so, so])
        return RHS_B

    def _get_E_2_U(self):
        A, B = self.A, self.B
        so, sv = self.so, self.sv
        E_2_U = 4 * np.einsum("Bpi, Api -> AB", B.U_1[:, :, so], A.nc_deriv.F_1_mo[:, :, so])
        E_2_U += 4 * np.einsum("Aai, Bai -> AB", A.U_1[:, sv, so], self.RHS_B)
        E_2_U += 4 * np.einsum("ABai, ai -> AB", self.pdB_B_A[:, :, sv, so], self.Z)
        E_2_U -= 2 * np.einsum("Aki, Bki -> AB", A.S_1_mo[:, so, so], B.pdA_nc_F_0_mo[:, so, so])
        E_2_U -= 2 * np.einsum("ABki, ki -> AB", self.pdB_S_A_mo[:, :, so, so], A.nc_deriv.F_0_mo[so, so])
        return E_2_U
