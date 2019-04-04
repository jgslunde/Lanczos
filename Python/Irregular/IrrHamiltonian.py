import numpy as np
import scipy.sparse
import os
from tqdm import tqdm
from IrrLap import Laplacian


class Hamiltonian:
    """ Class for setting up Hamiltonian. """
    def __init__(self, Grid, Potential, T_factor):
        self.Grid = Grid
        self.N = Grid.N
        self.L = Grid.L # Length of system in fm.
        self.Potential = Potential
        self.T_factor = T_factor


    def MakeSparseH(self):
        self.MakeSparseV()
        self.MakeSparseT()
        self.H_sparse = (-self.T_sparse + self.V_sparse)


    def MakeSparseV(self):
        print("+++ Setting up sparse potential matrix V.")
        N = self.N
        row_ind = []; col_ind = []; data = []
        for idx in tqdm(range(self.Grid.nr_gridpoints)):
            x, y, z = self.Grid(idx) - self.Grid.center
            row_ind.append(idx); col_ind.append(idx)
            data.append(self.Potential(x, y, z))
        self.V_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3, N**3))


    def MakeSparseT(self):
        Grid = self.Grid
        N = self.N
        T_factor = self.T_factor
        print("+++ Setting up sparse laplacian matrix T.")
        row_ind = []; col_ind = []; data = []
        for idx in tqdm(range(Grid.nr_gridpoints)):
            point = Grid.GridCoords[idx]
            neighbor_idxs = np.argwhere(np.linalg.norm(Grid.GridCoords - point, axis=1) < 3)
            # print(len(neighbor_idxs))
            # print(Grid.GridCoords[neighbor_idxs])
            neighbor_idxs = neighbor_idxs[neighbor_idxs != idx]  # Remove self from neighbors.
            # print(neighbor_idxs)
            # print(Grid.GridCoords[np.argsort(np.linalg.norm(Grid.GridCoords - point, axis=1))[:10]])
            # print(np.linalg.norm( Grid.GridCoords[np.argsort(np.linalg.norm(Grid.GridCoords - point, axis=1))[:10]] - point, axis=1))

            neighbor_points_relative = Grid.GridCoords[neighbor_idxs] - point
            weights = Laplacian(neighbor_points_relative)
            for i in range(len(neighbor_idxs)):
                neighbor = Grid.GridCoords[neighbor_idxs[i]]
                row_ind.append(idx)
                col_ind.append(neighbor_idxs[i])
                data.append(T_factor*weights[i])
        self.T_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))


        # filename = f"T_N={self.N}_Laplace={points}"
        # if os.path.isfile(f"T_matrices/{filename}.npz"):
        #     print(f"+++ Laplacian matrix T for N = {self.N} and {points} points already created. Extracting...")
        #     self.T_sparse = scipy.sparse.load_npz(f"T_matrices/{filename}.npz")
        # else:
        #     print(f"+++ Laplacian matrix T for N = {self.N} and {points} does not exist. Creating...")
        #     if points == "7":
        #         Laplacian = self.Laplacian_7point
        #     elif points =="27":
        #         Laplacian = self.Laplacian_27point
        #     T_factor, N = self.T_factor, self.N
        #     row_ind = []; col_ind = []; data = []
        #     for i in tqdm(range(N**3)):
        #         neighbors, weights = Laplacian(i)
        #         for j, neighbor in enumerate(neighbors):
        #             row_ind.append(i)
        #             col_ind.append(neighbor)
        #             data.append(T_factor*weights[j])
        #     self.T_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))
        #     scipy.sparse.save_npz(f"T_matrices/{filename}.npz", self.T_sparse)


if __name__ == "__main__":
    TEST = Hamiltonian(1,1,1,1)