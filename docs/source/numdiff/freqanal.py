import numpy as np
import scipy
from scipy.constants import physical_constants

E_h = physical_constants["Hartree energy"][0]
a_0 = physical_constants["Bohr radius"][0]
N_A = physical_constants["Avogadro constant"][0]
c_0 = physical_constants["speed of light in vacuum"][0]
e_c = physical_constants["elementary charge"][0]
e_0 = physical_constants["electric constant"][0]
mu_0 = physical_constants["mag. constant"][0]

class FreqAnal:
    
    def __init__(self, mol_weight, mol_coord, hessian):
        self.mol_weight = mol_weight
        self.natm = natm = self.mol_weight.size
        self.mol_coord = mol_coord
        self.mol_hess = hessian
        self.mol_hess = (self.mol_hess + self.mol_hess.T) / 2
        self.mol_hess = self.mol_hess.reshape((natm, 3, natm, 3))
        self._theta = NotImplemented     # Force constant tensor
        self._proj_inv = NotImplemented  # Inverse space of translation and rotation of theta
        self._freq = NotImplemented      # Frequency in cm-1 unit
        self._q = NotImplemented         # Unnormalized normal coordinate
        self._qnorm = NotImplemented     # Normalized normal coordinate
        
    @property
    def theta(self):
        if self._theta is NotImplemented:
            natm, mol_hess, mol_weight = self.natm, self.mol_hess, self.mol_weight
            self._theta = np.einsum("AtBs, A, B -> AtBs", mol_hess, 1 / np.sqrt(mol_weight), 1 / np.sqrt(mol_weight)).reshape(3 * natm, 3 * natm)
        return self._theta
    
    @property
    def center_coord(self):
        return (self.mol_coord * self.mol_weight[:, None]).sum(axis=0) / self.mol_weight.sum()
    
    @property
    def centered_coord(self):
        return self.mol_coord - self.center_coord
    
    @property
    def rot_eig(self):
        natm, centered_coord, mol_weight = self.natm, self.centered_coord, self.mol_weight
        rot_tmp = np.zeros((natm, 3, 3))
        rot_tmp[:, 0, 0] = centered_coord[:, 1]**2 + centered_coord[:, 2]**2
        rot_tmp[:, 1, 1] = centered_coord[:, 2]**2 + centered_coord[:, 0]**2
        rot_tmp[:, 2, 2] = centered_coord[:, 0]**2 + centered_coord[:, 1]**2
        rot_tmp[:, 0, 1] = rot_tmp[:, 1, 0] = - centered_coord[:, 0] * centered_coord[:, 1]
        rot_tmp[:, 1, 2] = rot_tmp[:, 2, 1] = - centered_coord[:, 1] * centered_coord[:, 2]
        rot_tmp[:, 2, 0] = rot_tmp[:, 0, 2] = - centered_coord[:, 2] * centered_coord[:, 0]
        rot_tmp = (rot_tmp * mol_weight[:, None, None]).sum(axis=0)
        _, rot_eig = np.linalg.eigh(rot_tmp)
        return rot_eig
    
    @property
    def proj_scr(self):
        natm, centered_coord, rot_eig, mol_weight = self.natm, self.centered_coord, self.rot_eig, self.mol_weight
        rot_coord = np.einsum("At, ts, rw -> Asrw", centered_coord, rot_eig, rot_eig)
        proj_scr = np.zeros((natm, 3, 6))
        proj_scr[:, (0, 1, 2), (0, 1, 2)] = 1
        proj_scr[:, :, 3] = (rot_coord[:, 1, :, 2] - rot_coord[:, 2, :, 1])
        proj_scr[:, :, 4] = (rot_coord[:, 2, :, 0] - rot_coord[:, 0, :, 2])
        proj_scr[:, :, 5] = (rot_coord[:, 0, :, 1] - rot_coord[:, 1, :, 0])
        proj_scr *= np.sqrt(mol_weight)[:, None, None]
        proj_scr.shape = (-1, 6)
        proj_scr /= np.linalg.norm(proj_scr, axis=0)
        return proj_scr
    
    @property
    def proj_inv(self):
        if self._proj_inv is NotImplemented:
            natm, proj_scr, theta = self.natm, self.proj_scr, self.theta
            proj_inv = np.zeros((natm * 3, natm * 3))
            proj_inv[:, :6] = proj_scr
            x_col = 6 - 1
            for A_col in range(6, natm * 3):
                stat = True
                while stat:  # first xcol - 6 values in vector t should be zero
                    x_col += 1
                    x0 = np.zeros((natm * 3, ))
                    x0[x_col - 6] = 1
                    t = x0 - proj_inv[:, :A_col].T @ x0 @ proj_inv[:, :A_col].T
                    t /= np.linalg.norm(t)
                    stat = (np.linalg.norm(t[:x_col - 6]) > 1e-7)
                proj_inv[:, A_col] = t
            proj_inv = proj_inv[:, 6:]
            self._proj_inv = proj_inv
        return self._proj_inv
    
    def _get_freq_qdiag(self):
        natm, proj_inv, theta, mol_weight = self.natm, self.proj_inv, self.theta, self.mol_weight
        e, q = np.linalg.eigh(proj_inv.T @ theta @ proj_inv)
        freq = np.sqrt(np.abs(e * E_h * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e > 0) * 2 - 1)
        self._freq = freq
        q_unnormed = np.einsum("AtQ, A -> AtQ", (proj_inv @ q).reshape(natm, 3, (proj_inv @ q).shape[-1]), 1 / np.sqrt(mol_weight))
        q_unnormed = q_unnormed.reshape(-1, q_unnormed.shape[-1])
        q_normed = q_unnormed / np.linalg.norm(q_unnormed, axis=0)
        self._q = q_unnormed
        self._qnorm = q_normed
    
    @property
    def freq(self):
        if self._freq is NotImplemented:
            self._get_freq_qdiag()
        return self._freq
    
    @property
    def q(self):
        if self._q is NotImplemented:
            self._get_freq_qdiag()
        return self._q
    
    @property
    def qnorm(self):
        if self._qnorm is NotImplemented:
            self._get_freq_qdiag()
        return self._qnorm