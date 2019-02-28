import numpy as np
import scipy.sparse
from tqdm import tqdm

class Hermitian:
    """ Class for setting up Hermitian. """
    def __init__(self, N, L, potential, T_factor):
        self.N = N
        self.L = L # Length of system in fm.
        self.potential = potential
        self.T_factor = T_factor
        self.dx = float(L)/N

        self.x = np.linspace(-L//2, L//2, N)
        self.y = np.linspace(-L//2, L//2, N)
        self.z = np.linspace(-L//2, L//2, N)



    def create_sparse_Hamiltonian(self):
        pass


    def create_sparse_V(self):
        print("+++ Setting up sparse potential matrix V.")
        N = self.N
        row_ind = []; col_ind = []; data = []
        for i in tqdm(range(N)):
           for j in range(N):
                for k in range(N):
                    idx = self.unravel_xyz(i, j, k)
                    row_ind.append(idx); col_ind.append(idx)
                    data.append(self.potential(self.x[i], self.y[j], self.z[k]))
        self.V_sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))


    def create_sparse_T(self):
        print("+++ Setting up sparse laplacian matrix T.")
        T_factor = self.T_factor
        N, stencil = self.N, self.stencil
        row_ind = []; col_ind = []; data = []
        for i in tqdm(range(N**3)):
            neighbors = stencil(i)
            for neighbor in neighbors:
                row_ind.append(i)
                col_ind.append(neighbor)
                data.append(T_factor)
            row_ind.append(i)
            col_ind.append(i)
            data.append(-6*T_factor)
        self.T_sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))




    def unravel_xyz(self, x, y, z):
        # Given x, y, z indexes, gives the unraveled index i.
        N = self.N
        return x + y*N + z*N**2

    def ravel_i(self, i):
        # Given the unraveled index i, gives the corresponding x,y,z indexes.
        N = self.N
        x = i%N
        y = (i//N)%N
        z = i//N**2
        return (x, y, z)

    def stencil(self, i):
        # Given an index i in the unraveled vector, returns the idexes for the 6 neighboring point, in prev/next, (x,y,z) order.
        ravel_i, unravel_xyz, N = self.ravel_i, self.unravel_xyz, self.N
        x, y, z = ravel_i(i)
        
        if x == 0:
            x_idx_prev, x_idx_next = unravel_xyz(N-1, y, z), unravel_xyz(x+1, y, z)
        elif x == N-1:
            x_idx_prev, x_idx_next = unravel_xyz(x-1, y, z), unravel_xyz(0, y, z)
        else:
            x_idx_prev, x_idx_next = unravel_xyz(x-1, y, z), unravel_xyz(x+1, y, z)

        if y == 0:
            y_idx_prev, y_idx_next = unravel_xyz(x, N-1, z), unravel_xyz(x, y+1, z)
        elif y == N-1:
            y_idx_prev, y_idx_next = unravel_xyz(x, y-1, z), unravel_xyz(x, 0, z)
        else:
            y_idx_prev, y_idx_next = unravel_xyz(x, y-1, z), unravel_xyz(x, y+1, z)

        if z == 0:
            z_idx_prev, z_idx_next = unravel_xyz(x, y, N-1), unravel_xyz(x, y, z+1)
        elif z == N-1:
            z_idx_prev, z_idx_next = unravel_xyz(x, y, z-1), unravel_xyz(x, y, 0)
        else:
            z_idx_prev, z_idx_next = unravel_xyz(x, y, z-1), unravel_xyz(x, y, z+1)

        return(x_idx_prev, x_idx_next, y_idx_prev, y_idx_next, z_idx_prev, z_idx_next)


    def test_stencil(self):
        # Tests that the stencil gives back the right coordinates for egde cases.
        N, stencil, unravel_xyz = self.N, self.stencil, self.unravel_xyz
        x, y, z = 0, 0, 0
        x_prev, x_next, y_prev, y_next, z_prev, z_next = stencil(unravel_xyz(x, y, z))
        assert (x_prev, x_next, y_prev, y_next, z_prev, z_next) == (N-1, 1, (N-1)*N, N, (N-1)*N**2, N**2)


    def Laplacian(self, i):
        # Returns the points with weights of the laplacian.
        pass

