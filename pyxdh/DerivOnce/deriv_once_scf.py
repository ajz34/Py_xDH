import numpy as np
from abc import ABC, abstractmethod
from functools import partial
import os
import warnings
import copy
from pyscf.scf._response_functions import _gen_rhf_response

from pyscf import gto, dft, grad, hessian
from pyscf.scf import cphf

from pyxdh.Utilities import timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: A1
class DerivOnceSCF(ABC):

    def __init__(self, config):

        # From configuration file, with default values
        self.config = config  # type: dict
        self.scf_eng = config["scf_eng"]  # type: dft.rks.RKS
        self.rotation = config.get("rotation", True)
        self.grdit_memory = config.get("grdit_memory", 2000)
        self.init_scf = config.get("init_scf", True)
        self.cphf_tol = config.get("cphf_tol", 1e-6)

        # Basic settings
        self.mol = self.scf_eng.mol  # type: gto.Mole
        self.scf_grad = NotImplemented
        self.scf_hess = NotImplemented
        if not self.rotation:
            warnings.warn("No orbital rotation can lead to fatal numerical discrepancy!")

        # Backdoor to DFT
        self.cx = 1
        self.xc = "HF"
        self.grids = NotImplemented  # type: dft.gen_grid.Grids
        self.xc_type = "HF"

        # From SCF Calculation
        self._C = NotImplemented
        self._mo_occ = NotImplemented
        self._e = NotImplemented
        self._D = NotImplemented
        self._F_0_ao = NotImplemented
        self._F_0_mo = NotImplemented
        self._H_0_ao = NotImplemented
        self._H_0_mo = NotImplemented
        self._eng = NotImplemented

        # From gradient and hessian calculation
        self._H_1_ao = NotImplemented
        self._H_1_mo = NotImplemented
        self._S_1_ao = NotImplemented
        self._S_1_mo = NotImplemented
        self._F_1_ao = NotImplemented
        self._F_1_mo = NotImplemented
        self._B_1 = NotImplemented
        self._U_1 = NotImplemented
        self._U_1_vo = NotImplemented
        self._U_1_ov = NotImplemented

        # Tensor total derivative
        self._pdA_F_0_mo = NotImplemented

        # ERI
        self._eri0_ao = NotImplemented
        self._eri0_mo = NotImplemented
        self._eri1_ao = NotImplemented
        self._eri1_mo = NotImplemented

        # Response generator
        self._resp = NotImplemented
        self._resp_cphf = NotImplemented

        # E1
        self._E_1 = NotImplemented

        # Initializer
        self.initialization()
        self.cphf_grids = config.get("cphf_grids", self.grids)

        return

    # region Initializers

    def initialization(self):
        self.initialization_pyscf()
        self.initialization_scf()

    def initialization_pyscf(self):
        if (self.scf_eng.mo_coeff is NotImplemented or self.scf_eng.mo_coeff is None) and self.init_scf:
            self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        if isinstance(self.scf_eng, dft.rks.RKS):
            self.xc = self.scf_eng.xc
            self.grids = self.scf_eng.grids
            self.xc_type = dft.libxc.xc_type(self.xc)
            self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
            self.scf_grad = grad.rks.Gradients(self.scf_eng)
            self.scf_hess = hessian.rks.Hessian(self.scf_eng)
        else:
            self.scf_grad = grad.RHF(self.scf_eng)
            self.scf_hess = hessian.RHF(self.scf_eng)
        return

    def initialization_scf(self):
        if self.init_scf:
            self.mo_occ = self.scf_eng.mo_occ
            self.C = self.scf_eng.mo_coeff
            self.e = self.scf_eng.mo_energy
        return

    # endregion

    # region Properties

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        if self._C is NotImplemented:
            self._C = C
        else:
            raise AttributeError("Once orbital coefficient is set, it should not be changed anymore.")

    @property
    def Co(self):
        return self.C[:, self.so]

    @property
    def Cv(self):
        return self.C[:, self.sv]

    @property
    def nmo(self):
        if self.C is NotImplemented:
            raise ValueError("Molecular orbital number should be determined after SCF process!\nPrepare self.C first.")
        return self.C.shape[1]

    @property
    def nao(self):
        return self.mol.nao

    @property
    def nocc(self):
        return self.mol.nelec[0]

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def mo_occ(self):
        return self._mo_occ

    @mo_occ.setter
    def mo_occ(self, mo_occ):
        if self._mo_occ is NotImplemented:
            self._mo_occ = mo_occ
        else:
            raise AttributeError("Once mo_occ is set, it should not be changed anymore.")

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

    @e.setter
    def e(self, e):
        if self._e is NotImplemented:
            self._e = e
        else:
            raise AttributeError("Once orbital energy is set, it should not be changed anymore.")

    @property
    def eo(self):
        return self.e[self.so]

    @property
    def ev(self):
        return self.e[self.sv]

    @property
    def D(self):
        if self._D is NotImplemented:
            self._D = self._get_D()
        return self._D

    @property
    def eng(self):
        if self._eng is NotImplemented:
            self._eng = self._get_eng()
        return self._eng

    @property
    def H_0_ao(self):
        if self._H_0_ao is NotImplemented:
            self._H_0_ao = self._get_H_0_ao()
        return self._H_0_ao

    @property
    def H_0_mo(self):
        if self._H_0_mo is NotImplemented:
            self._H_0_mo = self._get_H_0_mo()
        return self._H_0_mo

    @property
    def F_0_ao(self):
        if self._F_0_ao is NotImplemented:
            self._F_0_ao = self._get_F_0_ao()
        return self._F_0_ao

    @property
    def F_0_mo(self):
        if self._F_0_mo is NotImplemented:
            self._F_0_mo = self._get_F_0_mo()
        return self._F_0_mo

    @property
    def eri0_ao(self):
        warnings.warn("eri0_ao: ERI should not be stored in memory! Consider J/K engines!")
        if self._eri0_ao is NotImplemented:
            self._eri0_ao = self._get_eri0_ao()
        return self._eri0_ao

    @property
    def eri0_mo(self):
        warnings.warn("eri0_mo: ERI AO -> MO is quite expensive!")
        if self._eri0_mo is NotImplemented:
            self._eri0_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)
        return self._eri0_mo

    @property
    def H_1_ao(self):
        if self._H_1_ao is NotImplemented:
            self._H_1_ao = self._get_H_1_ao()
        return self._H_1_ao

    @property
    def H_1_mo(self):
        if self._H_1_mo is NotImplemented:
            self._H_1_mo = self._get_H_1_mo()
        return self._H_1_mo

    @property
    def F_1_ao(self):
        if self._F_1_ao is NotImplemented:
            self._F_1_ao = self._get_F_1_ao()
        return self._F_1_ao

    @property
    def F_1_mo(self):
        if self._F_1_mo is NotImplemented:
            self._F_1_mo = self._get_F_1_mo()
        return self._F_1_mo

    @property
    def S_1_ao(self):
        if self._S_1_ao is NotImplemented:
            self._S_1_ao = self._get_S_1_ao()
        return self._S_1_ao

    @property
    def S_1_mo(self):
        if self._S_1_mo is NotImplemented:
            self._S_1_mo = self._get_S_1_mo()
        return self._S_1_mo

    @property
    def eri1_ao(self):
        warnings.warn("eri1_ao: 4-idx tensor ERI should be not used!")
        if self._eri1_ao is NotImplemented:
            self._eri1_ao = self._get_eri1_ao()
        return self._eri1_ao

    @property
    def eri1_mo(self):
        warnings.warn("eri1_mo: 4-idx tensor ERI should be not used!")
        if self._eri1_mo is NotImplemented:
            self._eri1_mo = self._get_eri1_mo()
        return self._eri1_mo

    @property
    def B_1(self):
        if self._B_1 is NotImplemented:
            self._B_1 = self._get_B_1()
        return self._B_1

    @property
    def U_1_vo(self):
        return self.U_1[:, self.sv, self.so]

    @property
    def U_1_ov(self):
        return self.U_1[:, self.so, self.sv]

    @property
    def U_1(self):
        if self._U_1 is NotImplemented:
            self._U_1 = self._get_U_1()
        return self._U_1

    @property
    def E_1(self):
        if self._E_1 is NotImplemented:
            self._E_1 = self._get_E_1()
        return self._E_1

    @property
    def pdA_F_0_mo(self):
        if self._pdA_F_0_mo is NotImplemented:
            self._pdA_F_0_mo = self._get_pdA_F_0_mo()
        return self._pdA_F_0_mo

    @property
    def resp(self):
        if self._resp is NotImplemented:
            self._resp = _gen_rhf_response(self.scf_eng, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)
        return self._resp

    @property
    def resp_cphf(self):
        if self._resp_cphf is NotImplemented:
            if self.xc_type == "HF":
                self._resp_cphf = self.resp
            else:
                mf = dft.RKS(self.mol)
                mf.xc = self.scf_eng.xc
                mf.grids = self.cphf_grids
                self._resp_cphf = _gen_rhf_response(mf, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)
        return self._resp_cphf

    # endregion

    # region Utility functions

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    def Ax0_Core(self, si, sj, sk, sl, reshape=True, in_cphf=False):
        """

        Parameters
        ----------
        si : slice or None
        sj : slice or None
            ``si`` and ``sj`` should be all slice or all None. If chosen None, then return an AO base Ax(X).

        sk : slice or None
        sl : slice or None
            ``sk`` and ``sk`` should be all slice or all None. If chosen None, then `X` that passed in is assumed to
            be a density matrix.
        reshape : bool
        in_cphf : bool
            if ``in_cphf``, use ``self.cphf_grids`` instead of usual grid.

        Returns
        -------
        fx : function which pass matrix, then return Ax @ X.
        """
        C = self.C
        nao = self.nao
        resp = self.resp_cphf if in_cphf else self.resp

        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        @timing
        def fx(X_):
            if not isinstance(X_, np.ndarray):
                return 0
            X = X_.copy()  # type: np.ndarray
            shape1 = list(X.shape)
            X.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dm = X
                if dm.shape[-2] != nao or dm.shape[-1] != nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that X passed in is an AO-based matrix!")
            else:
                dm = C[:, sk] @ X @ C[:, sl].T
            dm += dm.transpose((0, 2, 1))

            ax_ao = resp(dm) * 2

            # Old Code (maybe not suitable for parallel Aaibj Ubj ......)
            # Use PySCF higher functions to avoid explicit eri0_ao storage
            # ax_ao = np.empty((dm.shape[0], nao, nao))
            # for idx, dmX in enumerate(dm):
            #     ax_ao[idx] = (
            #         + 1 * self.scf_eng.get_j(dm=dmX)
            #         - 0.5 * cx * self.scf_eng.get_k(dm=dmX)
            #     )
            # # GGA part
            # if self.xc_type == "GGA":
            #     grdit = GridIterator(self.mol, grids, self.D, deriv=2, memory=self.grdit_memory)
            #     for grdh in grdit:
            #         kerh = KernelHelper(grdh, self.xc)
            #         for idx, dmX in enumerate(dm):
            #             tmp_K = np.einsum("kl, gl -> gk", dmX, grdh.ao_0)
            #             rho_X_0 = np.einsum("gk, gk -> g", grdh.ao_0, tmp_K)
            #             rho_X_1 = 2 * np.einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
            #             gamma_XD = np.einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
            #             tmp_M = np.empty((4, grdh.ngrid))
            #             tmp_M[0] = (
            #                     np.einsum("g, g -> g", rho_X_0, kerh.frr)
            #                     + 2 * np.einsum("g, g -> g", gamma_XD, kerh.frg)
            #             )
            #             tmp_M[1:4] = (
            #                     + 4 * np.einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
            #                     + 8 * np.einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
            #                     + 4 * np.einsum("rg, g -> rg", rho_X_1, kerh.fg)
            #             )
            #             ax_ao[idx] += (
            #                 + np.einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
            #             )
            # ax_ao += ax_ao.swapaxes(-1, -2)

            if not sij_none:
                ax_ao = np.einsum("Auv, ui, vj -> Aij", ax_ao, C[:, si], C[:, sj])
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(ax_ao.shape[-2])
                shape1.append(ax_ao.shape[-1])
                ax_ao.shape = shape1
            return ax_ao

        return fx

    @abstractmethod
    def Ax1_Core(self, si, sj, sk, sl, reshape=True):
        pass

    # endregion

    # region Getting Functions

    def _get_D(self):
        return 2 * self.Co @ self.Co.T

    def _get_eng(self):
        return self.scf_eng.e_tot

    def _get_H_0_ao(self):
        return self.scf_eng.get_hcore()

    def _get_H_0_mo(self):
        return self.C.T @ self.H_0_ao @ self.C

    def _get_F_0_ao(self):
        return self.scf_eng.get_fock(dm=self.D)

    def _get_F_0_mo(self):
        return self.C.T @ self.F_0_ao @ self.C

    def _get_eri0_ao(self):
        return self.mol.intor("int2e")

    @abstractmethod
    def _get_H_1_ao(self):
        pass

    def _get_H_1_mo(self):
        if not isinstance(self.H_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, up, vq -> Apq", self.H_1_ao, self.C, self.C)

    @abstractmethod
    def _get_F_1_ao(self):
        pass

    def _get_F_1_mo(self):
        if not isinstance(self.F_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, up, vq -> Apq", self.F_1_ao, self.C, self.C)

    @abstractmethod
    def _get_S_1_ao(self):
        pass

    def _get_S_1_mo(self):
        if not isinstance(self.S_1_ao, np.ndarray):
            return 0
        return np.einsum("Auv, up, vq -> Apq", self.S_1_ao, self.C, self.C)

    @abstractmethod
    def _get_eri1_ao(self):
        pass

    @timing
    def _get_eri1_mo(self):
        if not isinstance(self.eri1_ao, np.ndarray):
            return 0
        return np.einsum("Auvkl, up, vq, kr, ls -> Apqrs", self.eri1_ao, self.C, self.C, self.C, self.C)

    def _get_B_1(self):
        sa = self.sa
        so = self.so

        B_1 = self.F_1_mo.copy()
        if isinstance(self.S_1_mo, np.ndarray):
            B_1 += (
                - self.S_1_mo * self.e
                - 0.5 * self.Ax0_Core(sa, sa, so, so)(self.S_1_mo[:, so, so])
            )
        return B_1

    def _get_U_1(self):
        B_1 = self.B_1
        S_1_mo = self.S_1_mo
        if not isinstance(S_1_mo, np.ndarray):
            S_1_mo = np.zeros_like(B_1)
        Ax0_Core = self.Ax0_Core
        sv = self.sv
        so = self.so

        # Generate v-o block of U
        U_1_ai = cphf.solve(
            self.Ax0_Core(sv, so, sv, so, in_cphf=True),
            self.e,
            self.scf_eng.mo_occ,
            B_1[:, sv, so],
            max_cycle=100,
            tol=self.cphf_tol,
            hermi=False
        )[0]
        U_1_ai.shape = (B_1.shape[0], self.nvir, self.nocc)

        # Test whether converged
        conv = (
            + U_1_ai * (self.ev[:, None] - self.eo[None, :])
            + self.Ax0_Core(sv, so, sv, so)(U_1_ai)
            + self.B_1[:, sv, so]
        )
        if abs(conv).max() > 1e-8:
            msg = "\nget_E_1: CP-HF not converged well!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)

        if self.rotation:
            # Generate rotated U
            U_1_pq = - 0.5 * S_1_mo
            U_1_pq[:, sv, so] = U_1_ai
            U_1_pq[:, so, sv] = - S_1_mo[:, so, sv] - U_1_pq[:, sv, so].swapaxes(-1, -2)
        else:
            # Generate total U
            D_pq = - (self.e[:, None] - self.e[None, :]) + 1e-300
            U_1_pq = np.zeros((B_1.shape[0], self.nmo, self.nmo))
            U_1_pq[:, sv, so] = U_1_ai
            U_1_pq[:, so, sv] = - S_1_mo[:, so, sv] - U_1_pq[:, sv, so].swapaxes(-1, -2)
            U_1_pq[:, so, so] = (Ax0_Core(so, so, sv, so)(U_1_ai) + B_1[:, so, so]) / D_pq[so, so]
            U_1_pq[:, sv, sv] = (Ax0_Core(sv, sv, sv, so)(U_1_ai) + B_1[:, sv, sv]) / D_pq[sv, sv]
            for p in range(self.nmo):
                U_1_pq[:, p, p] = - S_1_mo[:, p, p] / 2
            U_1_pq -= (U_1_pq + U_1_pq.swapaxes(-1, -2) + S_1_mo) / 2
            U_1_pq -= (U_1_pq + U_1_pq.swapaxes(-1, -2) + S_1_mo) / 2

        self._U_1 = U_1_pq
        return self._U_1

    @abstractmethod
    def _get_E_1(self):
        pass

    def _get_pdA_F_0_mo(self):
        F_1_mo = self.F_1_mo
        U_1 = self.U_1
        e = self.e
        Ax0_Core = self.Ax0_Core
        so, sa = self.so, self.sa

        pdA_F_0_mo = (
            + F_1_mo
            + np.einsum("Apq, p -> Apq", U_1, e)
            + np.einsum("Aqp, q -> Apq", U_1, e)
            + Ax0_Core(sa, sa, sa, so)(U_1[:, :, so])
        )
        return pdA_F_0_mo

    # endregion


# Cubic Inheritance: B1
class DerivOnceNCDFT(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceNCDFT, self).__init__(config)
        config_nc = copy.copy(config)
        config_nc["scf_eng"] = config_nc["nc_eng"]
        config_nc["init_scf"] = False
        self.nc_deriv = self.DerivOnceMethod(config_nc)
        self.nc_deriv.C = self.C
        self.nc_deriv.mo_occ = self.mo_occ
        self._Z = NotImplemented
        self._pdA_nc_F_0_mo = NotImplemented

    @property
    @abstractmethod
    def DerivOnceMethod(self):
        pass

    @property
    def Z(self):
        if self._Z is NotImplemented:
            self._Z = self._get_Z()
        return self._Z

    @property
    def pdA_nc_F_0_mo(self):
        if self._pdA_nc_F_0_mo is NotImplemented:
            self._pdA_nc_F_0_mo = self._get_pdA_nc_F_0_mo()
        return self._pdA_nc_F_0_mo

    def _get_Z(self):
        so, sv = self.so, self.sv
        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        F_0_mo = self.nc_deriv.F_0_mo
        Z = cphf.solve(Ax0_Core(sv, so, sv, so, in_cphf=True), e, mo_occ, F_0_mo[sv, so], max_cycle=100, tol=self.cphf_tol)[0]
        return Z

    def _get_pdA_nc_F_0_mo(self):
        nc_F_0_mo = self.nc_deriv.F_0_mo
        nc_F_1_mo = self.nc_deriv.F_1_mo
        U_1 = self.U_1
        Ax0_Core = self.nc_deriv.Ax0_Core
        so, sa = self.so, self.sa

        pdA_nc_F_0_mo = (
            + nc_F_1_mo
            + np.einsum("Amp, mq -> Apq", U_1, nc_F_0_mo)
            + np.einsum("Amq, pm -> Apq", U_1, nc_F_0_mo)
            + Ax0_Core(sa, sa, sa, so)(U_1[:, :, so])
        )
        return pdA_nc_F_0_mo

    def _get_eng(self):
        return self.nc_deriv.scf_eng.energy_tot(dm=self.D)
