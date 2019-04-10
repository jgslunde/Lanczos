import numpy as np
import matplotlib.pyplot as plt
import math as m
import time
from tqdm import trange
from itertools import product
from tools import unravel_ijk, ravel_idx
from symetry import FindMirrorSymetricPoints
from Potentials import Deuterium3DPotential

# Pre-calculate every possible neighboring displacement in 3 or 6 dimensions.
displacements3D = np.array(list(product([-1,0,1], repeat=3)))
displacements6D = np.array(list(product([-1,0,1], repeat=6)))
displacements3D_2 = np.array(list(product([-2,-1,0,1,2], repeat=3)))
displacements3D_3 = np.array(list(product([-3,-2,-1,0,1,2,3], repeat=3)))


class IrrGrid:

    class Box:
        def __init__(self, box_nr, idx_start, a, N, corner_coord):
            if N%a != 0:
                raise ValueError(f"Nr of fine grid points N={N} in each box must be multiple of a={a}.")
            self.box_nr = box_nr
            self.idx_start = idx_start
            self.a = a  # Fine grid spacing between points.
            self.N = N  # Nr fine grid points in each dir.
            self.n = N//a  # Nr actual grid points in each dir.
            self.nr_points = self.n**3
            self.corner_coord = corner_coord  # Coordinate of box corner in units s. Correpsonds to point idx 0.
            self.local_point_idxs = np.arange(0, self.n**3)  # Internal indexes of points in box, idx=0 correponding to box corner.
            self.point_idxs = self.local_point_idxs + idx_start  # Global indexes of points, idx=0 correponding to global lattice corner(and box 0 corner).
            self.point_coords = np.array([[i,j,k] for k in range(self.n) for j in range(self.n) for i in range(self.n)], dtype=int)*a + corner_coord  # Coordinate of each point, in units of s.
            self.point_coords_local = np.array([[i,j,k] for k in range(self.n) for j in range(self.n) for i in range(self.n)], dtype=int)  # Local coordinate of each point, in units of it's own grid size, a.
            self.neighbors = []  # Box numbers of neighbors, in established order.
            self.neighbor_displacements = []
            self.neighbor_disp_dict = {}
            self.neighbor_disp_dict_reversed = {}



    def __init__(self, N, L, d=3):
        self.N = N
        self.L = L
        self.potential_center = L/2.0



    def GetNearbyPoints(self, idx, D):
        # Return the indexes and relative positions to the points within a grid distance D of idx point (in units of a, the local box grid size).
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.BoxList[box_nr]
        a = box.a

        if not self.IsCloseToEdge(idx, D):  # If nearby points will only include points in current box, stuffs gets easy.
            if D == 1:
                disp = displacements3D
            elif D == 2:
                disp = displacements3D_2
            elif D == 3:
                disp = displacements3D_3
            else:
                raise ValueError("Displacements larger than 3 not yet implemented.")

            coord = box.point_coords_local[idx-box.idx_start]  # Note that these coords are not in units of local grid spacing a, not s.
            coords = coord + disp
            idxs = unravel_ijk(*coords.T, box.n) + box.idx_start

        else:  # If nearby points crosses into another box, but it has equal spacing, so the grid should be uniform.
            if D == 1:
                disp = displacements3D
            elif D == 2:
                disp = displacements3D_2
            elif D == 3:
                disp = displacements3D_3
            else:
                raise ValueError("Displacements larger than 3 not yet implemented.")

            idxs = np.zeros(disp.shape[0], dtype=int)
            coord = box.point_coords_local[idx-box.idx_start]
            coords = coord + disp
            neighbor_disp = (coords > box.n-1).astype(int) - (coords < 0).astype(int)

            for i in range(disp.shape[0]):
                if (neighbor_disp[i] == 0).all(): # If this point is in our box.
                    idxs[i] = unravel_ijk(*coords[i], box.n) + box.idx_start
                else:
                    neighbor_box_nr = box.neighbor_disp_dict_reversed[neighbor_disp[i].tobytes()]
                    neighbor_box = self.BoxList[neighbor_box_nr]
                    neighbor_coord = coords[i].copy() - neighbor_disp[i]*box.n
                    neighbor_coord = (neighbor_coord*box.a)//neighbor_box.a
                    idxs[i] = unravel_ijk(*neighbor_coord, neighbor_box.n) + neighbor_box.idx_start
        return np.unique(idxs)



    def IsCloseToEdge(self, idx, D, d=3):
        # Returns True if idx is within D steps from a neighboring box, and False otherwise.
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.BoxList[box_nr]
        coords = np.array(ravel_idx(idx - box.idx_start, box.n))  # Calculate the "internal" i,j,k, where 0,0,0 corresponds to box corner.
        disp = displacements3D.copy()*D
        neighbor_disp = (coords + disp > box.n-1).astype(int) - (coords + disp < 0).astype(int)
        return len(np.unique(neighbor_disp, axis=0)) > 1


    def IsCloseToEdgeWithDifferentSpacing(self, idx, D, d=3):
        # Returns True if idx is withing D steps from a neighboring box with different spacing than its own, and False otherwise.
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.BoxList[box_nr]
        coords = np.array(ravel_idx(idx - box.idx_start, box.n))  # Calculate the "internal" i,j,k, where 0,0,0 corresponds to box corner.
        disp = displacements3D.copy()*D
        neighbor_disp = (coords + disp > box.n-1).astype(int) - (coords + disp < 0).astype(int)
        neighbor_disp = np.unique(neighbor_disp, axis=0)
        print(neighbor_disp)
        print(self.point_coords[idx], box.N)

        for x in neighbor_disp:
            if (x != 0).any():
                neighbor_nr = box.neighbor_disp_dict_reversed[x.tobytes()]
                if self.aList[neighbor_nr] != box.a:
                    return True
        return False



    def Get27PointStencil(self, idx):
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.BoxList[box_nr]
        i0, j0, k0 = ravel_idx(idx - box.idx_start, box.n)  # Calculate the "internal" i,j,k, where 0,0,0 corresponds to box corner.
        print(box_nr, idx, box.idx_start)
        points = np.empty(27, dtype=int)
        points[:] = np.nan

        # Finding whatever neighboring box has the heighest spacing value.
        # From our given point, do a 1-step walk in every direction, and save all unique new boxes we end up in:
        disp = [[(i+i0>box.n-1).astype(int)-(i+i0<0).astype(int), (j+j0>box.n-1).astype(int)-(j+j0<0).astype(int), (k+k0>box.n-1).astype(int)-(k+k0<0).astype(int)] for k in range(-1,2) for j in range(-1,2) for i in range(-1,2)]
        disp = np.array(disp)
        disp = np.unique(disp, axis=0)

        # From these boxes we ended up in, find the one with the highest point spacing value:
        boxes_to_consider = [box_nr]
        for i in range(len(disp)):
            disp_bytes = disp[i].tobytes()
            if disp_bytes in box.neighbor_disp_dict_reversed:
                boxes_to_consider.append(box.neighbor_disp_dict_reversed[disp[i].tobytes()])
        a = np.max(self.aList[boxes_to_consider])  # Global a used for all relevant boxes, equal to the highest a of all relevant boxes.


        counter = 0
        jump = a//box.a
        # Now, considering the chosen spacing (which might be from a neighboring box), let's go a step of size a in each direction.
        for k in range(-a, a+1, a):
            for j in range(-a, a+1, a):
                for i in range(-a, a+1, a):
                    x, y, z = jump*i+i0, jump*j+j0, jump*k+k0
                    if 0 <= x <= box.n-1 and 0 <= y <= box.n-1 and 0 <= z <= box.n-1:  # If point lies within current box.
                        points[counter] = unravel_ijk(x, y, z, box.n) + box.idx_start
                        counter += 1
                        disp = (np.array([i,j,k]) > box.n-1).astype(int) - (np.array([i,j,k]) < 0).astype(int)
                    else:
                        # If point is outside box, find out in which direction. We create a "box displacement vector", [x,y,z], with x,y,z in {-1,0,1}.
                        out_of_box_disp = (np.array([x, y, z]) > box.n-1).astype(int) - (np.array([x, y, z]) < 0).astype(int)
                        neighbor_nr = box.neighbor_disp_dict_reversed[out_of_box_disp.tobytes()]  # The box nr we just stepped into.
                        # Getting the point from the neighboring box:
                        neighbor_box = self.BoxList[neighbor_nr]

                        if i < 0:
                            i = neighbor_box.n - i
                        elif i > box.n-1:
                            i = i - box.n
                        if j < 0:
                            j = neighbor_box.n - i
                        elif j > box.n-1:
                            j = j - box.n
                        if k < 0:
                            k = neighbor_box.n - i
                        elif k > box.n-1:
                            k = k - box.n

                        boxes_a_ratio = neighbor_box.a//box.a
                        if ((i0+i*jump)%boxes_a_ratio == 0) and ((j0+j*jump)%boxes_a_ratio == 0) and ((k0+k*jump)%boxes_a_ratio == 0):
                            i, j, k = i//boxes_a_ratio, j//boxes_a_ratio, k//boxes_a_ratio
                            points[counter] = unravel_ijk(x, y, z, neighbor_box.n) + neighbor_box.idx_start

        return points




    def CalculatePointDensity(self, box_corners, box_depth, Potential=Deuterium3DPotential):
        print("+++ Calculating grid density in every box...")
        nr_boxes = box_depth**3
        E0 = np.array([-1.626, 10.286])
        E_list = np.zeros(nr_boxes)
        for i in trange(nr_boxes):
            box_corner = box_corners[i]
            x, y, z = np.meshgrid(np.linspace(0, self.L/box_depth, 101), np.linspace(0, self.L/box_depth, 101), np.linspace(0, self.L/box_depth, 101))
            x += box_corner[0]-self.potential_center
            y += box_corner[1]-self.potential_center
            z += box_corner[2]-self.potential_center
            pot = Potential(x, y, z)
            E_list[i] = np.max([np.max(np.abs(pot - E0[0])), np.max(np.abs(pot - E0[1]))])
        a_factor = np.max(np.sqrt(E_list))/np.sqrt(E_list)
        for i in range(nr_boxes):
            print(i, a_factor[i])

        # a is set to the (rounded down) nearest 2*N value, unless it exceeds 1/8 the number of fine-grid points in each box,
        # in which case it is set to that (to avoid any box having less than 8 points).
        self.aList = 2**np.ceil(np.log(a_factor)/np.log(2))
        self.aList = np.array([min(int(x), self.N_per_box//8) for x in self.aList])
        # self.aList = np.array([round(x) for x in a_factor], dtype=int)
        print(f"+++ Grid density calculation finished. Density ranges from {np.min(self.aList)} to {np.max(self.aList)}")



    def SetupBoxes(self, box_depth):
        if self.N%box_depth != 0:
            raise ValueError(f"Number of fine grid points N={self.N} must be multiple of number of boxes in each direction, {box_depth}")
        else:
            self.N_per_box = self.N//box_depth

        box_corners = np.array([[i,j,k] for k in range(box_depth) for j in range(box_depth) for i in range(box_depth)])*self.N//box_depth

        self.nr_boxes = box_depth**3
        self.CalculatePointDensity(box_corners, box_depth)
        self.nr_points = np.sum((self.N_per_box//self.aList)**3, dtype=int)
        self.get_box_nr_from_idx = np.zeros(self.nr_points, dtype=int)
        self.point_coords = np.zeros((self.nr_points, 3), dtype=int)

        self.BoxList = []
        current_nr_points = 0
        for i in range(box_depth**3):
            box = self.Box(i, current_nr_points, self.aList[i], self.N_per_box, box_corners[i])
            self.BoxList.append(box)
            self.point_coords[current_nr_points : current_nr_points + box.nr_points] = box.point_coords
            self.get_box_nr_from_idx[current_nr_points : current_nr_points + box.nr_points] = i
            current_nr_points += box.nr_points

        displacements = np.array([[i,j,k] for k in range(-1,2) for j in range(-1,2) for i in range(-1,2)])
        for i in range(box_depth**3):

            box = self.BoxList[i]
            for j in range(27):
                disp = displacements[j]
                if (disp != 0).any():
                    neighbor_idx = self.FindRelativeIndex(i, box_depth, disp)
                    box.neighbors.append(neighbor_idx)
                    box.neighbor_displacements.append(disp)
                    # box.neighbor_disp_dict[j] = disp.tobytes()
                    box.neighbor_disp_dict_reversed[disp.tobytes()] = neighbor_idx



    def FindRelativeIndex(self, idx, N, disp):
        # Given an integer "idx", representing an unraveled index [x, y, z] on a [N,N,N] grid, return
        # the index corresponding to a displacement of "disp"=[Dx,Dy,Dz], aka [x+Dx,y+Dy,z+Dz].
        # Periodic boundary conditions are assumed.
        Di, Dj, Dk = disp
        i, j, k = ravel_idx(idx, N)
        i, j, k = i+Di, j+Dj, k+Dk
        if i == -1:
            i = N-1
        elif i == N:
            i = 0
        if j == -1:
            j = N-1
        elif j == N:
            j = 0
        if k == -1:
            k = N-1
        elif k == N:
            k = 0
        new_idx = unravel_ijk(i, j, k, N)
        return new_idx





def PlotStuff():
    N = 30
    L = 25
    box_depth = 3
    grid = IrrGrid(N, L)
    grid.SetupBoxes(box_depth=box_depth)


    idx = 2334
    center_coord = grid.point_coords[idx]
    print("COORDS:", center_coord)
    print(grid.IsCloseToEdge(idx, 1))
    print()
    t0 = time.time()
    neighbor_depth = 3
    neighbors_idxs = grid.GetNearbyPoints(idx, neighbor_depth)
    print(time.time() - t0)
    neighbors = grid.point_coords[neighbors_idxs]

    asdf = FindMirrorSymetricPoints(neighbors-center_coord)
    print(len(neighbors))
    print(len(asdf))

    print(neighbors)
    neighbors2D = neighbors[:,:2]
    fig, ax = plt.subplots(3,3, figsize=(20,20))
    for i in range(-neighbor_depth, neighbor_depth + 1):
        k = i + neighbor_depth
        z = center_coord[2] + int(i)
        print(z)
        for j in range(neighbors.shape[0]):
            if neighbors[j,2] == z:
                ax[k//3, k%3].scatter(*neighbors[j,:2], c="b")
                ax[k//3, k%3].set_title(f"Z = {z}")
                ax[k//3, k%3].set_xlim(center_coord[0]-neighbor_depth*2, center_coord[0]+neighbor_depth*2)
                ax[k//3, k%3].set_ylim(center_coord[1]-neighbor_depth*2, center_coord[1]+neighbor_depth*2)
        if i == 0:
            ax[k//3, k%3].scatter(*center_coord[:2], c="r")
        # plt.axhline(y=x*N//box_depth-0.5, ls="--", c="y")
        # plt.axvline(x=x*N//box_depth-0.5, ls="--", c="y")
    plt.show()






if __name__ == "__main__":
    N = 40
    L = 25
    box_depth = 10
    grid = IrrGrid(N, L)
    grid.SetupBoxes(box_depth=box_depth)
    pot = Deuterium3DPotential(np.linspace(0, 12.5, 1001), np.zeros(1001), np.zeros(1001))
    plt.plot(pot)
    plt.show()



    # plt.scatter(*neighbors2D.T)
    # plt.scatter(*grid.point_coords[idx])
    # plt.xlim(-1, N)
    # plt.ylim(-1, N)





    # idx = 1437
    # print(grid.point_coords[2549])
    # print(grid.IsCloseToEdge(idx, 1))

    # idxs = grid.GetNearbyPoints(idx, 1)
    # print(idxs)
    # print(grid.point_coords[idxs])
    # print(grid.point_coords[idx])
    # print(np.shape(grid.point_coords[idxs]))

    # x, y = np.meshgrid(np.linspace(0,14,15), np.linspace(0,14,15))

    # for i in range(27):
    #     box = grid.BoxList[i]
    #     print(box.nr_points)
    #     print(box.idx_start)


    # plt.pcolormesh(x, y, asdf[13,:,:])
    # plt.show()

    # print(asdf[13,1,0])
    # plt.plot(asdf[13,12,:])
    # plt.plot(asdf[13,13,:])
    # plt.show()

    # asdf = grid.Get27PointStencil(111999)
    # print(asdf)
    # x, y, z = grid.ravel_idx(asdf-40000, 20)
    # for i in range(27):
    #     print(x[i], y[i], z[i])
    # grid.PlotGrid()