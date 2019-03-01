from pyscf import gto, scf, lib
from pyscf.scf import cphf
import numpy as np

def my_hess_elec(scf_eng):
    
    #----- Definine variables and utilities
    
    # Define essential variable
    mol = scf_eng.mol
    nao = mol.nao
    nmo = scf_eng.mo_energy.shape[0]
    nocc = mol.nelec[0]
    natm = mol.natm
    C = scf_eng.mo_coeff
    Co = C[:, :nocc]
    e = scf_eng.mo_energy
    eo = e[:nocc]
    ev = e[nocc:]
    mo_occ = scf_eng.mo_occ
    D = scf_eng.make_rdm1()
    De = np.einsum("ui, i, vi -> uv", C, e * mo_occ, C)
    # grad-contrib integral
    int1e_ipovlp = mol.intor('int1e_ipovlp')
    int1e_ipkin = mol.intor("int1e_ipkin")
    int1e_ipnuc = mol.intor("int1e_ipnuc")
    int2e_ip1 = mol.intor("int2e_ip1")
    # hess-contrib integral
    int1e_ipipkin = mol.intor("int1e_ipipkin")
    int1e_ipkinip = mol.intor("int1e_ipkinip")
    int1e_ipipnuc = mol.intor("int1e_ipipnuc")
    int1e_ipnucip = mol.intor("int1e_ipnucip")
    int2e_ipip1 = mol.intor("int2e_ipip1")
    int2e_ipvip1 = mol.intor("int2e_ipvip1")
    int2e_ip1ip2 = mol.intor("int2e_ip1ip2")
    int1e_ipipovlp = mol.intor("int1e_ipipovlp")
    int1e_ipovlpip = mol.intor("int1e_ipovlpip")
    
    def mol_slice(atm_id):
        _, _, p0, p1 = mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)
    
    #----- Effective code : noU
    
    def get_hess_ao_noU_hcore(A, B):
        ao_matrix = np.zeros((3 * 3, nao, nao))
        zA, zB = mol.atom_charge(A), mol.atom_charge(B)
        sA, sB = mol_slice(A), mol_slice(B)
        if (A == B):
            ao_matrix[:, sA] += int1e_ipipkin[:, sA]
            ao_matrix[:, sA] += int1e_ipipnuc[:, sA]
            with mol.with_rinv_as_nucleus(A):
                ao_matrix -= zA * mol.intor('int1e_ipiprinv')
                ao_matrix -= zA * mol.intor('int1e_iprinvip')
        ao_matrix[:, sA, sB] += int1e_ipkinip[:, sA, sB]
        ao_matrix[:, sA, sB] += int1e_ipnucip[:, sA, sB]
        with mol.with_rinv_as_nucleus(B):
            ao_matrix[:, sA] += zB * mol.intor('int1e_ipiprinv')[:, sA]
            ao_matrix[:, sA] += zB * mol.intor('int1e_iprinvip')[:, sA]
        with mol.with_rinv_as_nucleus(A):
            ao_matrix[:, sB] += zA * mol.intor('int1e_ipiprinv')[:, sB]
            ao_matrix[:, sB] += zA * mol.intor('int1e_iprinvip').swapaxes(1, 2)[:, sB]
        ao_matrix += ao_matrix.swapaxes(1, 2)
        return ao_matrix
    
    eri_mat1 = np.einsum("Tuvkl, kl -> Tuv", int2e_ipip1, D) * 2 - np.einsum("Tukvl, kl -> Tuv", int2e_ipip1, D)
    eri_mat2 = np.einsum("Tuvkl, kl -> Tuv", int2e_ipvip1, D) * 2 - np.einsum("Tukvl, kl -> Tuv", int2e_ip1ip2, D)
    eri_tensor1 = int2e_ip1ip2 * 4 - int2e_ipvip1.swapaxes(2, 3) - int2e_ip1ip2.swapaxes(2, 4)
    
    def get_hess_ao_noU_eri(A, B):
        ao_matrix = np.zeros((9, nao, nao))
        sA, sB = mol_slice(A), mol_slice(B)
        if (A == B):
            ao_matrix[:, sA] += eri_mat1[:, sA]
        ao_matrix[:, sA, sB] += eri_mat2[:, sA, sB]
        ao_matrix[:, sA] += np.einsum("Tuvkl, kl -> Tuv", eri_tensor1[:, sA, :, sB], D[sB])
        return ao_matrix
    
    def get_hess_ao_noU_S(A, B):
        ao_matrix = np.zeros((9, nao, nao))
        sA, sB = mol_slice(A), mol_slice(B)
        if (A == B):
            ao_matrix[:, sA] -= int1e_ipipovlp[:, sA] * 2
        ao_matrix[:, sA, sB] -= int1e_ipovlpip[:, sA, sB] * 2
        return ao_matrix
    
    def get_hess_noU(A, B):
        return (np.einsum("Tuv, uv -> T", get_hess_ao_noU_hcore(A, B) + get_hess_ao_noU_eri(A, B), D).reshape(3, 3)
                + np.einsum("Tuv, uv -> T",  + get_hess_ao_noU_S(A, B), De).reshape(3, 3))
    
    hess_noU = np.array([ [ get_hess_noU(A, B) for B in range(natm) ] for A in range(natm) ])
    
    #----- Effective Code: with U
    
    def get_hess_ao_h1(A):
        ao_matrix = np.zeros((3, nao, nao))
        sA = mol_slice(A)
        ao_matrix[:, sA] = (- int1e_ipkin - int1e_ipnuc 
                            - np.einsum("tuvkl, kl -> tuv", int2e_ip1, D)
                            + 0.5 * np.einsum("tukvl, kl -> tuv", int2e_ip1, D)
                           )[:, sA]
        ao_matrix -= np.einsum("tkluv, kl -> tuv", int2e_ip1[:, sA], D[sA])
        ao_matrix += 0.5 * np.einsum("tkulv, kl -> tuv", int2e_ip1[:, sA], D[sA])
        with mol.with_rinv_as_nucleus(A):
            ao_matrix -= mol.intor("int1e_iprinv") * mol.atom_charge(A)
        return ao_matrix + ao_matrix.swapaxes(1, 2)

    def get_hess_ao_s1(A):
        ao_matrix = np.zeros((3, nao, nao))
        sA = mol_slice(A)
        ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
        return ao_matrix + ao_matrix.swapaxes(1, 2)
    
    def Ax(x):
        shape = x.shape
        x = x.reshape((-1, nmo, nocc))
        dx = C @ x @ Co.T
        v = np.zeros_like(x)
        for i in range(dx.shape[0]):
            v[i] = C.T @ scf_eng.get_veff(mol, dx[i] + dx[i].T) @ Co
        return 2 * v.reshape(shape)

    hess_ao_h1 = np.array([ get_hess_ao_h1(A) for A in range(natm) ])
    hess_ao_s1 = np.array([ get_hess_ao_s1(A) for A in range(natm) ])
    hess_pi_h1 = np.einsum("Atuv, up, vi -> Atpi", hess_ao_h1, C, Co)
    hess_pi_s1 = np.einsum("Atuv, up, vi -> Atpi", hess_ao_s1, C, Co)
    
    hess_U, hess_M = cphf.solve(Ax, e, mo_occ, hess_pi_h1.reshape(-1, nmo, nocc), hess_pi_s1.reshape(-1, nmo, nocc))
    hess_U.shape = (natm, 3, nmo, nocc); hess_M.shape = (natm, 3, nocc, nocc)
    
    hess_withU = 4 * np.einsum("Bspi, Atpi -> ABts", hess_U, hess_pi_h1)
    hess_withU -= 4 * np.einsum("Bspi, Atpi, i -> ABts", hess_U, hess_pi_s1, eo)
    hess_withU -= 2 * np.einsum("Atki, Bski -> ABts", hess_pi_s1[:, :, :nocc], hess_M)
    
    return hess_noU + hess_withU
    
def my_hess_nuc(scf_eng):
    
    mol = scf_eng.mol
    natm = mol.natm
    
    nuc_Z = np.einsum("M, N -> MN", mol.atom_charges(), mol.atom_charges())
    nuc_V = lib.direct_sum("Mt - Nt -> MNt", mol.atom_coords(), mol.atom_coords())
    nuc_rinv = 1 / (np.linalg.norm(nuc_V, axis=2) + np.diag([np.inf] * mol.natm))
    nuc_5 = 3 * np.einsum("AB, AB, ABt, ABs -> ABts", nuc_Z, nuc_rinv ** 5, nuc_V, nuc_V)
    nuc_3 = np.einsum("AB, AB -> AB", nuc_Z, nuc_rinv ** 3)
    mask_atm = np.eye(natm)[:, :, None, None]
    mask_3D = np.eye(3)[None, None, :, :]
    
    hess_nuc = np.zeros((natm, natm, 3, 3))
    hess_nuc -= nuc_5                                                        # ABts
    hess_nuc += nuc_5.sum(axis=1) [:, None, :, :]       * mask_atm           # ABts -> AAts
    hess_nuc += nuc_3             [:, :, None, None]    * mask_3D            # AB**
    hess_nuc -= nuc_3.sum(axis=1) [:, None, None, None] * mask_atm * mask_3D # AB** -> AA**
    
    return hess_nuc
    
if __name__ == '__main__':
    
    mol = gto.Mole()
    mol.atom = """
    O  0.0  0.0  0.0
    O  0.0  0.0  1.5
    H  1.0  0.0  0.0
    H  0.0  1.0  1.5
    """
    mol.basis = "6-31G"
    mol.build()

    scf_eng = scf.RHF(mol)
    scf_eng.kernel()
    
    my_hess = my_hess_elec(scf_eng) + my_hess_nuc(scf_eng)
    
    from pyscf import hessian
    
    scf_hess = hessian.RHF(scf_eng)
    pyscf_hess = scf_hess.kernel()

    print("Hessian correct:                ", np.allclose(my_hess, pyscf_hess))
    
    import time
    
    time0 = time.time()
    [my_hess_elec(scf_eng) + my_hess_nuc(scf_eng) for i in range(5)]
    time1 = time.time()
    print("Average time for my_hess:       ", (time1 - time0) / 5)
    
    time0 = time.time()
    [scf_hess.kernel() for i in range(5)]
    time1 = time.time()
    print("Average time for pyscf.hessian: ", (time1 - time0) / 5)
    