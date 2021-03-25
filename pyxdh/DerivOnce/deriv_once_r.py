# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# python utilities
from abc import ABC, abstractmethod
import warnings
import copy
# pyscf utilities
from pyscf.scf._response_functions import _gen_rhf_response
from pyscf import gto, dft, scf, lib, hessian
from pyscf.scf import cphf
# pyxdh utilities
from pyxdh.Utilities import timing, cached_property
# additional definition for hessian
scf.hf.RHF.Hessian = lib.class_as_method(hessian.rhf.Hessian)
dft.rks.RKS.Hessian = lib.class_as_method(hessian.rks.Hessian)

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
        self._nocc = NotImplemented  # type: int
        self._C = NotImplemented  # type: np.ndarray
        self._mo_occ = NotImplemented  # type: np.ndarray
        self._e = NotImplemented  # type: np.ndarray

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
        if isinstance(self.scf_eng, (dft.rks.RKS, dft.uks.UKS)):
            self.xc = self.scf_eng.xc
            self.grids = self.scf_eng.grids
            self.xc_type = dft.libxc.xc_type(self.xc)
            self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
            self.scf_grad = self.scf_eng.Gradients()
            self.scf_hess = self.scf_eng.Hessian()
        else:
            self.scf_grad = self.scf_eng.Gradients()
            self.scf_hess = self.scf_eng.Hessian()
        return

    def initialization_scf(self):
        if self.init_scf:
            self.mo_occ = self.scf_eng.mo_occ
            self.C = self.scf_eng.mo_coeff
            self.e = self.scf_eng.mo_energy
            self.nocc = self.mol.nelec[0]
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
    def nmo(self) -> int:
        if self.C is NotImplemented:
            raise ValueError("Molecular orbital number should be determined after SCF process!\nPrepare self.C first.")
        return self.C.shape[-1]

    @property
    def nao(self):
        return self.mol.nao

    @property
    def nocc(self):
        return self._nocc

    @nocc.setter
    def nocc(self, nocc):
        if self._nocc is NotImplemented:
            self._nocc = nocc
        else:
            raise AttributeError("Once occupation number is set, it should not be changed anymore.")

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
    def occ(self):
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

    @cached_property
    def D(self):
        return 2 * self.Co @ self.Co.T

    @cached_property
    def eng(self):
        return self.scf_eng.e_tot

    @cached_property
    def H_0_ao(self):
        return self.scf_eng.get_hcore()

    @cached_property
    def H_0_mo(self):
        return self.C.T @ self.H_0_ao @ self.C

    @cached_property
    def S_0_ao(self):
        return self.mol.intor("int1e_ovlp")

    @cached_property
    def S_0_mo(self):
        return self.C.T @ self.S_0_ao @ self.C

    @cached_property
    def F_0_ao(self):
        return self.scf_eng.get_fock(dm=self.D)

    @cached_property
    def F_0_mo(self):
        return self.C.T @ self.F_0_ao @ self.C

    @cached_property
    def eri0_ao(self):
        return self.mol.intor("int2e")

    @cached_property
    def eri0_mo(self):
        return einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)

    @cached_property
    @abstractmethod
    def H_1_ao(self):
        pass

    @cached_property
    def H_1_mo(self):
        if not isinstance(self.H_1_ao, np.ndarray):
            return 0
        return einsum("Auv, up, vq -> Apq", self.H_1_ao, self.C, self.C)

    @cached_property
    @abstractmethod
    def F_1_ao(self):
        pass

    @cached_property
    def F_1_mo(self):
        if not isinstance(self.F_1_ao, np.ndarray):
            return 0
        return einsum("Auv, up, vq -> Apq", self.F_1_ao, self.C, self.C)

    @cached_property
    @abstractmethod
    def S_1_ao(self):
        pass

    @cached_property
    def S_1_mo(self):
        if not isinstance(self.S_1_ao, np.ndarray):
            return 0
        return einsum("Auv, up, vq -> Apq", self.S_1_ao, self.C, self.C)

    @cached_property
    @abstractmethod
    def eri1_ao(self):
        pass

    @cached_property
    @timing
    def eri1_mo(self):
        if not isinstance(self.eri1_ao, np.ndarray):
            return 0
        return einsum("Auvkl, up, vq, kr, ls -> Apqrs", self.eri1_ao, self.C, self.C, self.C, self.C)

    @cached_property
    def B_1(self):
        sa = self.sa
        so = self.so

        B_1 = self.F_1_mo.copy()
        if isinstance(self.S_1_mo, np.ndarray):
            B_1 += (
                - self.S_1_mo * self.e
                - 0.5 * self.Ax0_Core(sa, sa, so, so)(self.S_1_mo[:, so, so])
            )
        return B_1

    @cached_property
    @timing
    def U_1(self):
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

        return U_1_pq

    @property
    def U_1_vo(self):
        return self.U_1[:, self.sv, self.so]

    @property
    def U_1_ov(self):
        return self.U_1[:, self.so, self.sv]

    @cached_property
    def E_1(self):
        return self._get_E_1()

    @abstractmethod
    def _get_E_1(self):
        pass

    @cached_property
    def pdA_F_0_mo(self):
        F_1_mo = self.F_1_mo
        U_1 = self.U_1
        e = self.e
        Ax0_Core = self.Ax0_Core
        so, sa = self.so, self.sa

        pdA_F_0_mo = (
            + F_1_mo
            + einsum("Apq, p -> Apq", U_1, e)
            + einsum("Aqp, q -> Apq", U_1, e)
            + Ax0_Core(sa, sa, sa, so)(U_1[:, :, so])
        )
        return pdA_F_0_mo

    @cached_property
    def resp(self):
        return _gen_rhf_response(self.scf_eng, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)

    @cached_property
    def resp_cphf(self):
        if self.xc_type == "HF":
            return self.resp
        else:
            mf = dft.RKS(self.mol)
            mf.xc = self.scf_eng.xc
            mf.grids = self.cphf_grids
            return _gen_rhf_response(mf, mo_coeff=self.C, mo_occ=self.mo_occ, hermi=1, max_memory=self.grdit_memory)

    # endregion

    # region Utility functions

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    def Ax0_Core(self, si, sj, sk, sl, reshape=True, in_cphf=False, C=None):
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
        if C is None:
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
            #             tmp_K = einsum("kl, gl -> gk", dmX, grdh.ao_0)
            #             rho_X_0 = einsum("gk, gk -> g", grdh.ao_0, tmp_K)
            #             rho_X_1 = 2 * einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
            #             gamma_XD = einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
            #             tmp_M = np.empty((4, grdh.ngrid))
            #             tmp_M[0] = (
            #                     einsum("g, g -> g", rho_X_0, kerh.frr)
            #                     + 2 * einsum("g, g -> g", gamma_XD, kerh.frg)
            #             )
            #             tmp_M[1:4] = (
            #                     + 4 * einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
            #                     + 8 * einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
            #                     + 4 * einsum("rg, g -> rg", rho_X_1, kerh.fg)
            #             )
            #             ax_ao[idx] += (
            #                 + einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
            #             )
            # ax_ao += ax_ao.swapaxes(-1, -2)

            if not sij_none:
                ax_ao = einsum("Auv, ui, vj -> Aij", ax_ao, C[:, si], C[:, sj])
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
        self.nc_deriv.nocc = self.nocc

    @property
    @abstractmethod
    def DerivOnceMethod(self):
        pass

    @cached_property
    def Z(self):
        so, sv = self.so, self.sv
        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        F_0_mo = self.nc_deriv.F_0_mo
        Z = cphf.solve(Ax0_Core(sv, so, sv, so, in_cphf=True), e, mo_occ, F_0_mo[sv, so], max_cycle=100, tol=self.cphf_tol)[0]
        return Z

    @cached_property
    def pdA_nc_F_0_mo(self):
        nc_F_0_mo = self.nc_deriv.F_0_mo
        nc_F_1_mo = self.nc_deriv.F_1_mo
        U_1 = self.U_1
        Ax0_Core = self.nc_deriv.Ax0_Core
        so, sa = self.so, self.sa

        pdA_nc_F_0_mo = (
            + nc_F_1_mo
            + einsum("Amp, mq -> Apq", U_1, nc_F_0_mo)
            + einsum("Amq, pm -> Apq", U_1, nc_F_0_mo)
            + Ax0_Core(sa, sa, sa, so)(U_1[:, :, so])
        )
        return pdA_nc_F_0_mo

    @cached_property
    def eng(self):
        return self.nc_deriv.scf_eng.energy_tot(dm=self.D)


# Cubic Inheritance: C1
class DerivOnceMP2(DerivOnceSCF, ABC):

    def __init__(self, config):
        super(DerivOnceMP2, self).__init__(config)
        self.cc = config.get("cc", 1.)
        self.os = config.get("os", 1.)
        self.ss = config.get("ss", 1.)

    # region Properties
    @cached_property
    def eng(self):
        return self.scf_eng.e_tot + (self.T_iajb * self.t_iajb * self.D_iajb).sum()

    @cached_property
    def D_iajb(self):
        return (
            + self.eo[:, None, None, None]
            - self.ev[None, :, None, None]
            + self.eo[None, None, :, None]
            - self.ev[None, None, None, :]
        )

    @cached_property
    def t_iajb(self):
        so, sv = self.so, self.sv
        return self.eri0_mo[so, sv, so, sv] / self.D_iajb

    @cached_property
    def T_iajb(self):
        return self.cc * ((self.os + self.ss) * self.t_iajb - self.ss * self.t_iajb.swapaxes(-1, -3))

    @cached_property
    def D_r_oovv(self):
        nmo = self.nmo
        so, sv = self.so, self.sv
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        D_r_oovv = np.zeros((nmo, nmo))
        D_r_oovv[so, so] = - 2 * einsum("iakb, jakb -> ij", T_iajb, t_iajb)
        D_r_oovv[sv, sv] = 2 * einsum("iajc, ibjc -> ab", T_iajb, t_iajb)
        return D_r_oovv

    @cached_property
    def L(self):
        return self._get_L()

    def _get_L(self):
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        eri0_mo = self.eri0_mo
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        L = np.zeros((nvir, nocc))
        L += Ax0_Core(sv, so, sa, sa)(self.D_r_oovv)
        L -= 4 * einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])
        L += 4 * einsum("ibjc, abjc -> ai", T_iajb, eri0_mo[sv, sv, so, sv])
        return L

    @cached_property
    def D_r(self):
        L = self.L
        D_r = np.copy(self.D_r_oovv)
        so, sv = self.so, self.sv
        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        D_r[sv, so] = cphf.solve(Ax0_Core(sv, so, sv, so, in_cphf=True), e, mo_occ, L, max_cycle=100, tol=self.cphf_tol)[0]
        conv = (
            + D_r[sv, so] * (self.ev[:, None] - self.eo[None, :])
            + Ax0_Core(sv, so, sv, so)(D_r[sv, so]) + L
        )
        if abs(conv).max() > 1e-8:
            msg = "\nget_E_1: CP-HF not converged well!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)
        return D_r

    @cached_property
    def W_I(self):
        so, sv = self.so, self.sv
        nmo = self.nmo
        T_iajb = self.T_iajb
        eri0_mo = self.eri0_mo
        W_I = np.zeros((nmo, nmo))
        W_I[so, so] = - 2 * einsum("iakb, jakb -> ij", T_iajb, eri0_mo[so, sv, so, sv])
        W_I[sv, sv] = - 2 * einsum("iajc, ibjc -> ab", T_iajb, eri0_mo[so, sv, so, sv])
        W_I[sv, so] = - 4 * einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])
        return W_I

    @cached_property
    def pdA_eri0_mo(self):
        eri0_mo = self.eri0_mo
        U_1 = self.U_1
        pdA_eri0_mo = (
            + self.eri1_mo
            + einsum("pjkl, Api -> Aijkl", eri0_mo, U_1)
            + einsum("ipkl, Apj -> Aijkl", eri0_mo, U_1)
            + einsum("ijpl, Apk -> Aijkl", eri0_mo, U_1)
            + einsum("ijkp, Apl -> Aijkl", eri0_mo, U_1)
        )
        return pdA_eri0_mo

    @cached_property
    def pdA_t_iajb(self):
        so, sv = self.so, self.sv
        D_iajb = self.D_iajb
        pdA_F_0_mo = self.pdA_F_0_mo
        t_iajb = self.t_iajb
        pdA_eri0_mo = self.pdA_eri0_mo
        pdA_t_iajb = (
            + pdA_eri0_mo[:, so, sv, so, sv]
            - einsum("Aki, kajb -> Aiajb", pdA_F_0_mo[:, so, so], t_iajb)
            - einsum("Akj, iakb -> Aiajb", pdA_F_0_mo[:, so, so], t_iajb)
            + einsum("Aca, icjb -> Aiajb", pdA_F_0_mo[:, sv, sv], t_iajb)
            + einsum("Acb, iajc -> Aiajb", pdA_F_0_mo[:, sv, sv], t_iajb)
        ) / D_iajb
        return pdA_t_iajb

    @cached_property
    def pdA_T_iajb(self):
        return self.cc * ((self.os + self.ss) * self.pdA_t_iajb - self.ss * self.pdA_t_iajb.swapaxes(-1, -3))

    @cached_property
    def pdA_D_r_oovv(self):
        so, sv = self.so, self.sv
        nmo = self.nmo

        pdB_D_r_oovv = np.zeros((self.pdA_t_iajb.shape[0], nmo, nmo))
        pdB_D_r_oovv[:, so, so] -= 2 * einsum("iakb, Ajakb -> Aij", self.T_iajb, self.pdA_t_iajb)
        pdB_D_r_oovv[:, sv, sv] += 2 * einsum("iajc, Aibjc -> Aab", self.T_iajb, self.pdA_t_iajb)
        pdB_D_r_oovv[:, so, so] -= 2 * einsum("Aiakb, jakb -> Aij", self.pdA_T_iajb, self.t_iajb)
        pdB_D_r_oovv[:, sv, sv] += 2 * einsum("Aiajc, ibjc -> Aab", self.pdA_T_iajb, self.t_iajb)

        return pdB_D_r_oovv

    @cached_property
    def pdA_W_I(self):
        so, sv = self.so, self.sv
        natm, nmo = self.natm, self.nmo
        pdA_T_iajb, T_iajb = self.pdA_T_iajb, self.T_iajb
        eri0_mo, pdA_eri0_mo = self.eri0_mo, self.pdA_eri0_mo

        pdR_W_I = np.zeros((natm * 3, nmo, nmo))
        pdR_W_I[:, so, so] -= 2 * einsum("Aiakb, jakb -> Aij", pdA_T_iajb, eri0_mo[so, sv, so, sv])
        pdR_W_I[:, sv, sv] -= 2 * einsum("Aiajc, ibjc -> Aab", pdA_T_iajb, eri0_mo[so, sv, so, sv])
        pdR_W_I[:, sv, so] -= 4 * einsum("Ajakb, ijbk -> Aai", pdA_T_iajb, eri0_mo[so, so, sv, so])
        pdR_W_I[:, so, so] -= 2 * einsum("iakb, Ajakb -> Aij", T_iajb, pdA_eri0_mo[:, so, sv, so, sv])
        pdR_W_I[:, sv, sv] -= 2 * einsum("iajc, Aibjc -> Aab", T_iajb, pdA_eri0_mo[:, so, sv, so, sv])
        pdR_W_I[:, sv, so] -= 4 * einsum("jakb, Aijbk -> Aai", T_iajb, pdA_eri0_mo[:, so, so, sv, so])

        return pdR_W_I

    # endregion


# Cubic Inheritance: D1
class DerivOnceXDH(DerivOnceMP2, DerivOnceNCDFT, ABC):

    def _get_L(self):
        L = super(DerivOnceXDH, self)._get_L()
        L += 4 * self.nc_deriv.F_0_mo[self.sv, self.so]
        return L

    @cached_property
    def eng(self):
        return self.nc_deriv.scf_eng.energy_tot(dm=self.D) + (self.T_iajb * self.t_iajb * self.D_iajb).sum()
