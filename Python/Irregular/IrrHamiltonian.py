import numpy as np
import scipy.sparse
import os
from tqdm import trange
from IrrLap import Laplacian
from tools import get_relative_positions

class Hamiltonian:
    """ Class for setting up Hamiltonian. """
    def __init__(self, Grid, Potential, T_factor, normal_eq=False):
        self.Grid = Grid
        self.N = Grid.N
        self.L = Grid.L # Length of system in fm.
        self.Potential = Potential
        self.T_factor = T_factor
        self.normal_eq = normal_eq


    def MakeSparseH(self):
        self.MakeSparseV()
        self.MakeSparseT()
        self.H_sparse = (-self.T_sparse + self.V_sparse)
        if self.normal_eq:
            self.H_sparse = self.H_sparse.transpose()*self.H_sparse


    def MakeSparseV(self):
        print("+++ Setting up sparse potential matrix V.")
        N = self.N
        row_ind = []; col_ind = []; data = []
        for idx in trange(self.Grid.nr_points):
            x, y, z = self.Grid.point_coords[idx]*self.Grid.s - self.Grid.potential_center
            row_ind.append(idx); col_ind.append(idx)
            data.append(self.Potential(x, y, z))
        self.V_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3, N**3))
        scipy.sparse.save_npz("irreg.npz", self.V_sparse)


    def MakeSparseT(self):
        Grid = self.Grid
        N = self.N
        print("+++ Setting up sparse laplacian matrix T.")
        T_factor = self.T_factor
        row_ind = []; col_ind = []; data = []
        for idx in trange(Grid.nr_points):
            point = Grid.point_coords[idx]
            neighbor_idxs = Grid.GetNearbyPoints(idx, 1)
            neighbor_idxs = neighbor_idxs[neighbor_idxs != idx]  # Remove self from neighbors.
            if len(neighbor_idxs) < 26:
                # print(f"WARNING: Only {len(neighbor_idxs)} neighbors found when constructing Laplacian for idx {idx}.")
                # print(f"Employing temp. solution to this problem: Expanding Search Radius by 1.")
                neighbor_idxs = Grid.GetNearbyPoints(idx, 2)
                neighbor_idxs = neighbor_idxs[neighbor_idxs != idx]  # Remove self from neighbors.

                # print(neighbor_idxs)
                # print(Grid.GridCoords[np.argsort(np.linalg.norm(Grid.GridCoords - point, axis=1))[:10]])
                # print(np.linalg.norm( Grid.GridCoords[np.argsort(np.linalg.norm(Grid.GridCoords - point, axis=1))[:10]] - point, axis=1))

                neighbor_points_relative = get_relative_positions(point, Grid.point_coords[neighbor_idxs], self.N)
                weights = Laplacian(neighbor_points_relative)
                
                row_ind.append(idx)
                col_ind.append(idx)
                data.append(-np.sum(weights)*T_factor)

                for i in range(len(neighbor_idxs)):
                    row_ind.append(idx)
                    col_ind.append(neighbor_idxs[i])
                    data.append(T_factor*weights[i])
            self.T_sparse = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N**3,N**3))