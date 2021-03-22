# basic utilities
import numpy as np
from opt_einsum import contract as einsum
# pyscf utilities
from pyscf.scf import _vhf
# pyxdh utilities
from pyxdh.DerivTwice import DerivTwiceSCF, DerivTwiceNCDFT, DerivTwiceMP2, DerivTwiceXDH
from pyxdh.Utilities import timing, GridIterator, KernelHelper, cached_property
# pytest
from pyscf import gto, scf, dft
from pyxdh.DerivOnce import GradSCF, GradNCDFT, GradMP2, GradXDH
from pkg_resources import resource_filename
from pyxdh.Utilities import FormchkInterface
import pickle


# Cubic Inheritance: A2
class HessSCF(DerivTwiceSCF):

    # Assert A and B are the same GradSCF instance
    @property
    def A_is_B(self) -> bool:
        return True

    @cached_property
    def H_2_ao(self):
        scf_hess = self.A.scf_hess
        dhess, nao = self.natm * 3, self.nao
        H_2_ao = np.array([[scf_hess.hcore_generator()(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return H_2_ao.swapaxes(1, 2).reshape((dhess, dhess, nao, nao))

    @cached_property
    def S_2_ao(self):
        int1e_ipovlpip = self.mol.intor("int1e_ipovlpip")
        int1e_ipipovlp = self.mol.intor("int1e_ipipovlp")

        def get_S_SS_ao(A, B):
            ao_matrix = np.zeros((9, self.nao, self.nao))
            sA = self.mol_slice(A)
            sB = self.mol_slice(B)
            ao_matrix[:, sA, sB] = int1e_ipovlpip[:, sA, sB]
            if A == B:
                ao_matrix[:, sA] += int1e_ipipovlp[:, sA]
            return (ao_matrix + ao_matrix.swapaxes(1, 2)).reshape(3, 3, self.nao, self.nao)

        dhess, nao = self.natm * 3, self.nao
        S_2_ao = np.array([[get_S_SS_ao(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return S_2_ao.swapaxes(1, 2).reshape((dhess, dhess, nao, nao))

    @cached_property
    @timing
    def F_2_ao_JKcontrib(self):
        D = self.D
        eri2_ao = self.eri2_ao
        return (
            einsum("ABuvkl, kl -> ABuv", eri2_ao, D),
            einsum("ABukvl, kl -> ABuv", eri2_ao, D),
        )

    @timing
    def _get_F_2_ao_JKcontrib_old(self):

        mol = self.mol
        natm = self.natm
        D = self.D
        nao = self.nao

        def reshape_only_first_dimension(mats_, d1=3, d2=3):
            if isinstance(mats_, np.ndarray):
                mats = [mats_]
            else:
                mats = mats_
            for mat in mats:
                s = list(mat.shape)
                s[0] = d2
                s.insert(0, d1)
                mat.shape = tuple(s)

        Jcontrib = np.zeros((natm, natm, 3, 3, nao, nao))
        Kcontrib = np.zeros((natm, natm, 3, 3, nao, nao))
        hbas = (0, mol.nbas)

        # Atom insensitive contractions
        j_1 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipip1'), "s2kl",
            ("lk->s1ij"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        j_2 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipvip1'), "s2kl",
            ("lk->s1ij"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        k_1 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipip1'), "s2kl",
            ("jk->s1il"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        k_3 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ip1ip2'), "s1",
            ("lj->s1ki"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )

        reshape_only_first_dimension((j_1, j_2, k_1, k_3))

        # One atom sensitive contractions, multiple usage
        j_3A, k_1A, k_2A, k_3A = [], [], [], []
        for A in range(natm):
            shl0A, shl1A, p0A, p1A = mol.aoslice_by_atom()[A]
            sA, hA = slice(p0A, p1A), (shl0A, shl1A)

            j_3A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ip1ip2'), "s1",
                ("lk->s1ij"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hbas + hA + hbas)
            ))
            k_1A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ipip1'), "s2kl",
                ("li->s1kj"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hA + hbas * 3)
            ))
            k_2A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ipvip1'), "s2kl",
                ("jk->s1il"),
                D[sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hA + hbas + hbas)
            ))
            k_3A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ip1ip2'), "s1",
                ("jk->s1il"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hbas + hA + hbas)
            ))
        for jkA in j_3A, k_1A, k_2A, k_3A:
            reshape_only_first_dimension(jkA)

        for A in range(natm):
            shl0A, shl1A, p0A, p1A = mol.aoslice_by_atom()[A]
            sA, hA = slice(p0A, p1A), (shl0A, shl1A)

            # One atom sensitive contractions, One usage only
            j_1A = _vhf.direct_mapdm(
                mol._add_suffix('int2e_ipip1'), "s2kl",
                ("ji->s1kl"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hA + (0, mol.nbas) * 3)
            )
            reshape_only_first_dimension((j_1A,))

            # A-A manipulation
            Jcontrib[A, A, :, :, sA, :] += j_1[:, :, sA, :]
            Jcontrib[A, A] += j_1A
            Kcontrib[A, A, :, :, sA] += k_1[:, :, sA]
            Kcontrib[A, A] += k_1A[A]

            for B in range(A + 1):
                shl0B, shl1B, p0B, p1B = mol.aoslice_by_atom()[B]
                sB, hB = slice(p0B, p1B), (shl0B, shl1B)

                # Two atom sensitive contractions
                j_2AB = _vhf.direct_mapdm(
                    mol._add_suffix('int2e_ipvip1'), "s2kl",
                    ("ji->s1kl"),
                    D[sB, sA], 9,
                    mol._atm, mol._bas, mol._env,
                    shls_slice=(hA + hB + (0, mol.nbas) * 2)
                )
                k_3AB = _vhf.direct_mapdm(
                    mol._add_suffix('int2e_ip1ip2'), "s1",
                    ("ki->s1jl"),
                    D[sB, sA], 9,
                    mol._atm, mol._bas, mol._env,
                    shls_slice=(hA + (0, mol.nbas) + hB + (0, mol.nbas))
                )
                reshape_only_first_dimension((j_2AB, k_3AB))

                # A-B manipulation
                Jcontrib[A, B, :, :, sA, sB] += j_2[:, :, sA, sB]
                Jcontrib[A, B] += j_2AB
                Jcontrib[A, B, :, :, sA] += 2 * j_3A[B][:, :, sA]
                Jcontrib[B, A, :, :, sB] += 2 * j_3A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA] += k_2A[B][:, :, sA]
                Kcontrib[B, A, :, :, sB] += k_2A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA] += k_3A[B][:, :, sA]
                Kcontrib[B, A, :, :, sB] += k_3A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA, sB] += k_3[:, :, sB, sA].swapaxes(-1, -2)
                Kcontrib[A, B] += k_3AB

            # A == B finalize

            Jcontrib[A, A] /= 2
            Kcontrib[A, A] /= 2

        # Symmetry Finalize
        Jcontrib += Jcontrib.transpose((0, 1, 2, 3, 5, 4))
        Jcontrib += Jcontrib.transpose((1, 0, 3, 2, 4, 5))
        Kcontrib += Kcontrib.transpose((0, 1, 2, 3, 5, 4))
        Kcontrib += Kcontrib.transpose((1, 0, 3, 2, 4, 5))

        dhess = natm * 3
        return (Jcontrib.swapaxes(1, 2).reshape((dhess, dhess, nao, nao)),
                Kcontrib.swapaxes(1, 2).reshape((dhess, dhess, nao, nao)))

    @cached_property
    @timing
    def F_2_ao_GGAcontrib(self):
        if self.xc_type != "GGA":
            return 0

        natm = self.natm
        nao = self.nao

        F_2_ao_GGA = np.zeros((natm, natm, 3, 3, nao, nao))

        grdit = GridIterator(self.mol, self.grids, self.D, deriv=3, memory=self.grdit_memory)
        for grdh in grdit:
            kerh = KernelHelper(grdh, self.xc, deriv=3)
            pd_fr = kerh.frr * grdh.A_rho_1 + kerh.frg * grdh.A_gamma_1
            pd_fg = kerh.frg * grdh.A_rho_1 + kerh.fgg * grdh.A_gamma_1
            pd_rho_1 = grdh.A_rho_2
            pd_frr = kerh.frrr * grdh.A_rho_1 + kerh.frrg * grdh.A_gamma_1
            pd_frg = kerh.frrg * grdh.A_rho_1 + kerh.frgg * grdh.A_gamma_1
            pd_fgg = kerh.frgg * grdh.A_rho_1 + kerh.fggg * grdh.A_gamma_1
            pdpd_fr = (
                    + einsum("Bsg, Atg -> ABtsg", pd_frr, grdh.A_rho_1)
                    + einsum("Bsg, Atg -> ABtsg", pd_frg, grdh.A_gamma_1)
                    + kerh.frr * grdh.AB_rho_2 + kerh.frg * grdh.AB_gamma_2
            )
            pdpd_fg = (
                    + einsum("Bsg, Atg -> ABtsg", pd_frg, grdh.A_rho_1)
                    + einsum("Bsg, Atg -> ABtsg", pd_fgg, grdh.A_gamma_1)
                    + kerh.frg * grdh.AB_rho_2 + kerh.fgg * grdh.AB_gamma_2
            )
            pdpd_rho_1 = grdh.AB_rho_3

            # Contrib 1
            contrib1 = (
                    + 0.5 * einsum("ABtsg, gu, gv -> ABtsuv", pdpd_fr, grdh.ao_0, grdh.ao_0)
                    + 2 * einsum("ABtsg, rg, rgu, gv -> ABtsuv", pdpd_fg, grdh.rho_1, grdh.ao_1, grdh.ao_0)
                    + 2 * einsum("Atg, Bsrg, rgu, gv -> ABtsuv", pd_fg, pd_rho_1, grdh.ao_1, grdh.ao_0)
                    + 2 * einsum("Bsg, Atrg, rgu, gv -> ABtsuv", pd_fg, pd_rho_1, grdh.ao_1, grdh.ao_0)
                    + 2 * einsum("g, ABtsrg, rgu, gv -> ABtsuv", kerh.fg, pdpd_rho_1, grdh.ao_1, grdh.ao_0)
            )
            contrib1 += contrib1.swapaxes(-1, -2)
            F_2_ao_GGA += contrib1

            # Contrib 2
            tmp_contrib = (
                    - einsum("Bsg, tgu, gv -> Btsuv", pd_fr, grdh.ao_1, grdh.ao_0)
                    - 2 * einsum("Bsg, rg, tgu, rgv -> Btsuv", pd_fg, grdh.rho_1, grdh.ao_1, grdh.ao_1)
                    - 2 * einsum("Bsg, rg, trgu, gv -> Btsuv", pd_fg, grdh.rho_1, grdh.ao_2, grdh.ao_0)
                    - 2 * einsum("g, Bsrg, tgu, rgv -> Btsuv", kerh.fg, pd_rho_1, grdh.ao_1, grdh.ao_1)
                    - 2 * einsum("g, Bsrg, trgu, gv -> Btsuv", kerh.fg, pd_rho_1, grdh.ao_2, grdh.ao_0)
            )
            contrib2 = np.zeros((natm, natm, 3, 3, nao, nao))
            for A in range(natm):
                sA = self.mol_slice(A)
                contrib2[A, :, :, :, sA] += tmp_contrib[:, :, :, sA]
            contrib2 += contrib2.transpose((0, 1, 2, 3, 5, 4))
            contrib2 += contrib2.transpose((1, 0, 3, 2, 4, 5))
            F_2_ao_GGA += contrib2

            # Contrib 3
            contrib3 = np.zeros((natm, natm, 3, 3, nao, nao))

            tmp_contrib = (
                    + einsum("g, tsgu, gv -> tsuv", kerh.fr, grdh.ao_2, grdh.ao_0)
                    + 2 * einsum("g, rg, tsrgu, gv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_3, grdh.ao_0)
                    + 2 * einsum("g, rg, tsgu, rgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
            )
            for A in range(natm):
                sA = self.mol_slice(A)
                contrib3[A, A, :, :, sA] += tmp_contrib[:, :, sA]
            tmp_contrib = (
                    + einsum("g, tgu, sgv -> tsuv", kerh.fr, grdh.ao_1, grdh.ao_1)
                    + 2 * einsum("g, rg, trgu, sgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
                    + 2 * einsum("g, rg, tgu, srgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_1, grdh.ao_2)
            )
            for A in range(natm):
                for B in range(natm):
                    sA, sB = self.mol_slice(A), self.mol_slice(B)
                    contrib3[A, B, :, :, sA, sB] += tmp_contrib[:, :, sA, sB]
            contrib3 += contrib3.swapaxes(-1, -2)
            F_2_ao_GGA += contrib3

        # Finalize
        dhess = natm * 3
        return F_2_ao_GGA.swapaxes(1, 2).reshape((dhess, dhess, nao, nao))

    @cached_property
    @timing
    def eri2_ao(self):
        natm = self.natm
        nao = self.nao
        mol_slice = self.mol_slice

        int2e_ipip1 = self.mol.intor("int2e_ipip1")
        int2e_ipvip1 = self.mol.intor("int2e_ipvip1")
        int2e_ip1ip2 = self.mol.intor("int2e_ip1ip2")

        def get_eri2(A, B):
            sA, sB = mol_slice(A), mol_slice(B)
            eri2 = np.zeros((9, nao, nao, nao, nao))

            if A == B:
                eri2[:, sA, :, :, :] += int2e_ipip1[:, sA]
                eri2[:, :, sA, :, :] += int2e_ipip1[:, sA].transpose(0, 2, 1, 3, 4)
                eri2[:, :, :, sA, :] += int2e_ipip1[:, sA].transpose(0, 3, 4, 1, 2)
                eri2[:, :, :, :, sA] += int2e_ipip1[:, sA].transpose(0, 3, 4, 2, 1)
            eri2[:, sA, sB, :, :] += int2e_ipvip1[:, sA, sB]
            eri2[:, sB, sA, :, :] += einsum("Tijkl -> Tjikl", int2e_ipvip1[:, sA, sB])
            eri2[:, :, :, sA, sB] += einsum("Tijkl -> Tklij", int2e_ipvip1[:, sA, sB])
            eri2[:, :, :, sB, sA] += einsum("Tijkl -> Tklji", int2e_ipvip1[:, sA, sB])
            eri2[:, sA, :, sB, :] += int2e_ip1ip2[:, sA, :, sB]
            eri2[:, sB, :, sA, :] += einsum("Tijkl -> Tklij", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, sA, :, :, sB] += einsum("Tijkl -> Tijlk", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, sB, :, :, sA] += einsum("Tijkl -> Tklji", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sA, sB, :] += einsum("Tijkl -> Tjikl", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sB, sA, :] += einsum("Tijkl -> Tlkij", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sA, :, sB] += einsum("Tijkl -> Tjilk", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sB, :, sA] += einsum("Tijkl -> Tlkji", int2e_ip1ip2[:, sA, :, sB])

            return eri2.reshape((3, 3, nao, nao, nao, nao))

        return np.array([[get_eri2(A, B) for B in range(natm)] for A in range(natm)])\
            .swapaxes(1, 2).reshape((natm * 3, natm * 3, nao, nao, nao, nao))

    @timing
    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):

        mol = self.mol
        mol_slice = self.mol_slice
        natm = self.natm
        D = self.D
        dhess = natm * 3

        if grids is None:
            grids = self.grids
        if xc is None:
            xc = self.xc
        if cx is None:
            cx = self.cx
        if xc_type is None:
            xc_type = self.xc_type

        # GGA Contribution
        E_SS_GGA_contrib1 = np.zeros((natm, natm, 3, 3))
        E_SS_GGA_contrib2 = np.zeros((natm, natm, 3, 3))
        E_SS_GGA_contrib3 = np.zeros((natm, natm, 3, 3))
        if xc_type == "GGA":
            grdit = GridIterator(mol, grids, D, deriv=3, memory=self.grdit_memory)
            for grdh in grdit:
                kerh = KernelHelper(grdh, xc)

                tmp_tensor_1 = (
                        + 2 * einsum("g, Tgu, gv -> Tuv", kerh.fr, grdh.ao_2T, grdh.ao_0)
                        + 4 * einsum("g, rg, rTgu, gv -> Tuv", kerh.fg, grdh.rho_1, grdh.ao_3T, grdh.ao_0)
                        + 4 * einsum("g, rg, Tgu, rgv -> Tuv", kerh.fg, grdh.rho_1, grdh.ao_2T, grdh.ao_1)
                )
                XX, XY, XZ, YY, YZ, ZZ = range(6)
                for A in range(natm):
                    sA = mol_slice(A)
                    E_SS_GGA_contrib1[A, A] += einsum("Tuv, uv -> T", tmp_tensor_1[:, sA], D[sA])[
                        [XX, XY, XZ, XY, YY, YZ, XZ, YZ, ZZ]].reshape(3, 3)

                tmp_tensor_2 = 4 * einsum("g, rg, trgu, sgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
                tmp_tensor_2 += tmp_tensor_2.transpose((1, 0, 3, 2))
                tmp_tensor_2 += 2 * einsum("g, tgu, sgv -> tsuv", kerh.fr, grdh.ao_1, grdh.ao_1)
                E_SS_GGA_contrib2_inbatch = np.zeros((natm, natm, 3, 3))
                for A in range(natm):
                    sA = mol_slice(A)
                    for B in range(A + 1):
                        sB = mol_slice(B)
                        E_SS_GGA_contrib2_inbatch[A, B] += einsum("tsuv, uv -> ts",
                                                                     tmp_tensor_2[:, :, sA, sB], D[sA, sB])
                        if A != B:
                            E_SS_GGA_contrib2_inbatch[B, A] += E_SS_GGA_contrib2_inbatch[A, B].T
                E_SS_GGA_contrib2 += E_SS_GGA_contrib2_inbatch

                E_SS_GGA_contrib3 += (
                        + einsum("g, Atg, Bsg -> ABts", kerh.frr, grdh.A_rho_1, grdh.A_rho_1)
                        + 2 * einsum("g, wg, Atwg, Bsg -> ABts", kerh.frg, grdh.rho_1, grdh.A_rho_2, grdh.A_rho_1)
                        + 2 * einsum("g, Atg, rg, Bsrg -> ABts", kerh.frg, grdh.A_rho_1, grdh.rho_1, grdh.A_rho_2)
                        + 4 * einsum("g, wg, Atwg, rg, Bsrg -> ABts", kerh.fgg, grdh.rho_1, grdh.A_rho_2, grdh.rho_1,
                                        grdh.A_rho_2)
                        + 2 * einsum("g, Atrg, Bsrg -> ABts", kerh.fg, grdh.A_rho_2, grdh.A_rho_2)
                )

        E_SS_GGA_contrib = E_SS_GGA_contrib1 + E_SS_GGA_contrib2 + E_SS_GGA_contrib3
        E_SS_GGA_contrib = E_SS_GGA_contrib.swapaxes(1, 2).reshape((dhess, dhess))

        # HF Contribution
        E_SS_HF_contrib = (
                + einsum("ABuv, uv -> AB", self.H_2_ao, D)
                + 0.5 * einsum("ABuv, uv -> AB", self.F_2_ao_Jcontrib - 0.5 * cx * self.F_2_ao_Kcontrib, D)
        )

        E_SS = E_SS_GGA_contrib + E_SS_HF_contrib
        return E_SS

    def _get_E_2(self):
        dhess = self.natm * 3
        return self.E_2_Skeleton + self.E_2_U + self.A.scf_hess.hess_nuc().swapaxes(1, 2).reshape((dhess, dhess))


class HessNCDFT(DerivTwiceNCDFT, HessSCF):

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        if grids is None:
            grids = self.A.nc_deriv.grids
        if xc is None:
            xc = self.A.nc_deriv.xc
        if cx is None:
            cx = self.A.nc_deriv.cx
        if xc_type is None:
            xc_type = self.A.nc_deriv.xc_type
        return HessSCF._get_E_2_Skeleton(self, grids, xc, cx, xc_type)


# Cubic Inheritance: C2
class HessMP2(DerivTwiceMP2, HessSCF):
    pass


# Cubic Inheritance: D2
class HessXDH(DerivTwiceXDH, HessMP2, HessNCDFT):

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return HessNCDFT._get_E_2_Skeleton(self, grids, xc, cx, xc_type)


class TestHessR:

    mol = gto.Mole(atom="N 0. 0. 0.; H 1.5 0. 0.2; H 0.1 1.2 0.; H 0. 0. 1.", basis="6-31G", verbose=0).build()
    grids = dft.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.build()
    grids_cphf = dft.Grids(mol)
    grids_cphf.atom_grid = (50, 194)
    grids_cphf.build()

    def test_r_rhf_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        scf_hess = scf_eng.Hessian().run()
        gradh = GradSCF({"scf_eng": scf_eng})
        hessh = HessSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-HF-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)
        # ASSERT: hessian - PySCF
        assert np.allclose(hessh.E_2, scf_hess.de.swapaxes(-2, -3).reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_b3lyp_hess(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        scf_hess = scf_eng.Hessian().run()
        gradh = GradSCF({"scf_eng": scf_eng, "cphf_grids": self.grids_cphf})
        hessh = HessSCF({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-B3LYP-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-5, rtol=2e-4)
        # ASSERT: hessian - PySCF
        assert np.allclose(hessh.E_2, scf_hess.de.swapaxes(-2, -3).reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_hfb3lyp_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        nc_eng = dft.RKS(self.mol, xc="B3LYPg")
        nc_eng.grids = self.grids
        gradh = GradNCDFT({"scf_eng": scf_eng, "nc_eng": nc_eng})
        hessh = HessNCDFT({"deriv_A": gradh})
        with open(resource_filename("pyxdh", "Validation/numerical_deriv/NH3-HFB3LYP-hess.dat"), "rb") as f:
            ref_grad = pickle.load(f)
        # ASSERT: hessian - numerical
        assert np.allclose(hessh.E_2, ref_grad.reshape((-1, self.mol.natm * 3)), atol=1e-6, rtol=1e-4)

    def test_r_mp2_hess(self):
        scf_eng = scf.RHF(self.mol).run()
        gradh = GradMP2({"scf_eng": scf_eng})
        hessh = HessMP2({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-MP2-freq.fchk"))
        # ASSERT: hessian - Gaussian
        assert np.allclose(hessh.E_2, formchk.hessian(), atol=1e-6, rtol=1e-4)

    def test_r_xyg3_hess(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.3211, "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        hessh = HessXDH({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYG3-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(hessh.E_2, formchk.hessian(), atol=2e-5, rtol=2e-4)

    def test_r_xygjos_grad(self):
        scf_eng = dft.RKS(self.mol, xc="B3LYPg"); scf_eng.grids = self.grids; scf_eng.run()
        nc_eng = dft.RKS(self.mol, xc="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"); nc_eng.grids = self.grids
        config = {"scf_eng": scf_eng, "nc_eng": nc_eng, "cc": 0.4364, "ss": 0., "cphf_grids": self.grids_cphf}
        gradh = GradXDH(config)
        hessh = HessXDH({"deriv_A": gradh})
        formchk = FormchkInterface(resource_filename("pyxdh", "Validation/gaussian/NH3-XYGJOS-freq.fchk"))
        # ASSERT: hessian - Gaussian
        np.allclose(hessh.E_2, formchk.hessian(), atol=2e-5, rtol=2e-4)

