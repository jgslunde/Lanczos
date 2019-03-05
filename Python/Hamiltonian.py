import numpy as np
import scipy.sparse
from tqdm import tqdm

class Hamiltonian:
    """ Class for setting up Hamiltonian. """
    def __init__(self, N, L, potential, T_factor):
        self.N = N
        self.L = L # Length of system in fm.
        self.potential = potential
        self.T_factor = T_factor
        self.dx = float(L)/N

        self.x = np.linspace(-L//2, L//2, N)
        self.y = np.linspace(-L//2, L//2, N)
        self.z = np.linspace(-L//2, L//2, N)

        # Setting up 7-point stencil and weights.
        self.neighbors_relative_7point = np.array([[0,0,0], [-1,0,0], [0,-1,0], [0,0,-1], [1,0,0], [0,1,0], [0,0,1]])
        self.weights_7point = np.ones(7); self.weights_7point[0] = -6

        # Setting up 27-point stencil and weights.
        self.neighbors_relative_27point = np.array([[i,j,k] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2)])
        self.weights_27point = self.get_weights_27point()


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
        self.V_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))


    def create_sparse_T(self):
        print("+++ Setting up sparse laplacian matrix T.")
        T_factor, N = self.T_factor, self.N
        row_ind = []; col_ind = []; data = []
        for i in tqdm(range(N**3)):
            neighbors, weights = self.Laplacian_27point(i)
            for j, neighbor in enumerate(neighbors):
                row_ind.append(i)
                col_ind.append(neighbor)
                data.append(T_factor*weights[j])
            #row_ind.append(i)
            #col_ind.append(i)
            #data.append(-6*T_factor*weights[i])
        self.T_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))




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


    def Laplacian_7point(self, i):
        # Returns the points with weights of the laplacian.
        x, y, z = self.ravel_i(i)
        neighbors_relative = self.neighbors_relative_7point
        neighbors = neighbors_relative + np.array([x,y,z])
        for neighbor in neighbors:
            for i in range(3):
                if neighbor[i] == self.N:
                    neighbor[i] = 0
                elif neighbor[i] == -1:
                    neighbor[i] = self.N-1
        neighbors_i = [self.unravel_xyz(x,y,z) for x,y,z in neighbors]
        return neighbors_i, self.weights_7point


    def Laplacian_27point(self, i):
        x, y, z = self.ravel_i(i)
        neighbors_relative = self.neighbors_relative_27point
        neighbors = neighbors_relative + np.array([x,y,z])
        for neighbor in neighbors:
            for i in range(3):
                if neighbor[i] == self.N:
                    neighbor[i] = 0
                elif neighbor[i] == -1:
                    neighbor[i] = self.N-1
        neighbors_i = [self.unravel_xyz(x,y,z) for x,y,z in neighbors]
        return neighbors_i, self.weights_27point


    def get_weights_27point(self):
        weights_27point = np.zeros(27)
        for i in range(27):
            if (self.neighbors_relative_27point[i] == 0).all():  # Center
                weights_27point[i] = -44/3
            elif (self.neighbors_relative_27point[i] != 0).all():  # Corners
                weights_27point[i] = 1.0/3
            elif ( np.sum(self.neighbors_relative_27point[i] != 0) > 1):  # Edge
                weights_27point[i] = 1.0/2
            else:
                weights_27point[i] = 1.0
        weights_27point = weights_27point*3.0/13
        return weights_27point


if __name__ == "__main__":
    TEST = Hamiltonian(1,1,1,1)