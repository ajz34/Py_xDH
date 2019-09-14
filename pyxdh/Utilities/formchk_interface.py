import numpy as np


class FormchkInterface:

    def __init__(self, file_path):
        self.file_path = file_path
        self.natm = NotImplemented
        self.nao = NotImplemented
        self.nmo = NotImplemented
        self.initialization()

    def initialization(self):
        self.natm = int(self.key_to_value("Number of atoms"))
        self.nao = int(self.key_to_value("Number of basis functions"))
        self.nmo = int(self.key_to_value("Number of independent functions"))

    def key_to_value(self, key, file_path=None):
        if file_path is None:
            file_path = self.file_path
        flag_read = False
        expect_size = -1
        vec = []
        with open(file_path, "r") as file:
            for l in file:
                if l[:len(key)] == key:
                    try:
                        expect_size = int(l[len(key):].split()[2])
                        flag_read = True
                        continue
                    except IndexError:
                        try:
                            return float(l[len(key):].split()[1])
                        except IndexError:
                            continue
                if flag_read:
                    try:
                        vec += [float(i) for i in l.split()]
                    except ValueError:
                        break
        if len(vec) != expect_size:
            raise ValueError("Number of expected size is not consistent with read-in size!")
        return np.array(vec)

    def total_energy(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Total Energy", file_path)

    def grad(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Cartesian Gradient", file_path).reshape((self.natm, 3))

    def dipole(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Moment", file_path)

    @staticmethod
    def tril_to_symm(tril: np.ndarray):
        dim = int(np.floor(np.sqrt(tril.size * 2)))
        if dim * (dim + 1) / 2 != tril.size:
            raise ValueError("Size " + str(tril.size) + " is probably not a valid lower-triangle matrix.")
        indices_tuple = np.tril_indices(dim)
        iterator = zip(*indices_tuple)
        symm = np.empty((dim, dim))
        for it, (row, col) in enumerate(iterator):
            symm[row, col] = tril[it]
            symm[col, row] = tril[it]
        return symm

    def hessian(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.tril_to_symm(self.key_to_value("Cartesian Force Constants", file_path))

    def polarizability(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        return self.tril_to_symm(self.key_to_value("Polarizability  ", file_path))

    def dipolederiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Derivatives", file_path).reshape(-1, 3)
