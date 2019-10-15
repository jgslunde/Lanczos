import numpy as np
from tools2 import unravel, ravel, get_combinations, get_displaced_index
from Stencils import Stencils

class Lattice:

    class Box:
        def __init__(self, box_nr, start_idx_global, a_local, N_fine, corner_coord, nr_dims):
            self.nr_dims = nr_dims
            self.box_nr = box_nr  # Unique global numbering of boxes.
            self.start_idx_global = start_idx_global  # Global index of first point in box.
            self.a_local = a_local  # Local spacing in box, in units of finegrid spacing.
            self.N_fine = N_fine  # Number of finegrid points in each dim.
            self.N_local = N_fine//a_local  # Number of local gridpoints in each dim.
            self.nr_points = self.N_local**nr_dims  # Total number of lattice points (on local grid).
            self.corner_coord = corner_coord  # Coordinate of box corner, where point with idx 0 is located, in units of finegrid spacing s.
            self.local_point_idxs = np.arange(0, self.N_local**nr_dims, dtype=int)  # Local indexes of points. Idx 0 is at corner_coord, and corresponds to global idx_start.
            self.global_point_idxs = self.local_point_idxs + start_idx_global  # Unique, global index of points in box.
            self.coords_local = get_combinations(0, self.N_local-1, self.nr_dims)  # Local coordinates of points in box, in idxs order, in units of local spacing, a.
            self.coords_global = self.coords_local*self.a_local + self.corner_coord  # Above, but in global frame.

            self.neighbors = []  # List of box_nr for neighboring boxes, in established order. To be filled.
            self.neighbor_displacements = []  # List of vectors giving relative displacement in each dim of neighboring boxes, i.e. [-1, 0, -1].
            self.neighbor_disp_dict = {}  # Reverse of above. Dictionary holding the displacements of neighbors as (binary) keys, and their idxs as elements.

            self.test_constraints()


        def test_constraints(self):
            # Tests some chosen constrains for legal box spacing.
            assert self.N_fine%self.a_local == 0, f"Number of finegrid points, {self.N_fine}, must be multiple of local spacing, {self.a_local}."
            assert ((self.a_local & (self.a_local - 1)) == 0) and self.a_local > 0, f"Local spacing, {self.a}, must be power of 2, and positive."


        def get_points_in_cube(self, searchcube):
            # From [d,2] array searchcube representing the outer edges of a cube in local fine-grid units, return the index of all gridpoints inside.
            nr_dims = searchcube.shape[0]
            searchcube_l = searchcube//self.a_local   # Establish a search cube in local grid coordinates by dividing by the grid spacing.
            searchcube_l[:,0] += searchcube[:,0]%self.a_local != 0   # The upper limits were correctly rounded down, but the lower ones are supposed to be rouneded up,
                                                          # so we add +1 if they were in fact rounded down.

            idxs = []
            if nr_dims == 3:
                for i in range(searchcube_l[0,0], searchcube_l[0,1]+1):
                    for j in range(searchcube_l[1,0], searchcube_l[1,1]+1):
                        for k in range(searchcube_l[2,0], searchcube_l[2,1]+1):
                            idxs.append(unravel((i, j, k), self.N_local) + self.start_idx_global)

            if nr_dims == 6:
                for i in range(searchcube_l[0,0], searchcube_l[0,1]+1):
                    for j in range(searchcube_l[1,0], searchcube_l[1,1]+1):
                        for k in range(searchcube_l[2,0], searchcube_l[2,1]+1):
                            for x in range(searchcube_l[3,0], searchcube_l[3,1]+1):
                                for y in range(searchcube_l[4,0], searchcube_l[4,1]+1):
                                    for z in range(searchcube_l[5,0], searchcube_l[5,1]+1):
                                        idxs.append(unravel((i, j, k, x, y, z), self.N_local) + self.start_idx_global)

            if nr_dims == 2:
                for i in range(searchcube_l[0,0], searchcube_l[0,1]+1):
                    for j in range(searchcube_l[1,0], searchcube_l[1,1]+1):
                        idxs.append(unravel((i,j), self.N_local) + self.start_idx_global)

            return idxs


    def __init__(self, N, L, nr_dims=3):
        self.N = N  # Total number of fine-grid points in each dimension.
        self.L = L  # Length of lattice in each dim, in units of fm.
        self.nr_dims = nr_dims
        self.s = L/(N-1)  # Finegrid spacing, in units of fm.
        self.coords = []  # Coordinates of lattice points, in units of s.
        self.Box_list = []  # List containing class instances of all boxes.
        self.a_list = []  # List of grid spacings, a, in all boxes.
        self.box_corners = []  # Coordinates of corners of boxes, at local idx 0 (and smallest global idx of box).
        self.nr_points = -1  # Total number of (actual, not fine grid) points in lattice.
        self.get_box_nr_from_idx = []  # List containing box number of a given point idx.

        self.stencils = Stencils(self)

    def get_stencil(self, idx, method="Naive"):
        # Returns a list of the indexes in the stencil of point idx.
        return self.stencils.calc_hash(idx)


    def setup_boxes(self, box_depth):
        """Setting up the boxes and relations between them. """
        # Checking some conditions.
        assert box_depth%2 != 0, f"Currently only accepting odd number of boxes."
        assert self.N%box_depth == 0, f"Number of boxes in each dim, {box_depth}, must be multiple of number of finegrid points, {self.N}."
        assert (self.N//box_depth)%2 == 0, f"Number of points per box, {self.N//box_depth}, must be even."

        # Calculating some quantities.
        self.N_per_box = self.N//box_depth
        self.nr_boxes = box_depth**self.nr_dims
        self.a_list = np.array(self.calculate_grid_spacings())
        self.nr_points = np.sum((self.N_per_box//self.a_list)**self.nr_dims, dtype=int)
        self.box_corners = get_combinations(0, box_depth-1, self.nr_dims)*self.N//box_depth
        self.coords = np.zeros((self.nr_points, self.nr_dims), dtype=int)

        # Adding all Box instances to the box-list of Lattice class.
        current_nr_points = 0
        self.get_box_nr_from_idx = np.zeros(self.nr_points, dtype=int)
        for i in range(box_depth**self.nr_dims):
            box = self.Box(i, current_nr_points, self.a_list[i], self.N_per_box, self.box_corners[i], self.nr_dims)
            self.Box_list.append(box)
            self.coords[current_nr_points : current_nr_points + box.nr_points] = box.coords_global
            self.get_box_nr_from_idx[current_nr_points : current_nr_points + box.nr_points] = i
            current_nr_points += box.nr_points

        displacements = get_combinations(-1, 1, self.nr_dims)
        for i in range(box_depth**self.nr_dims):
            box = self.Box_list[i]
            for j in range(3**self.nr_dims):
                disp = displacements[j]
                neighbor_idx = get_displaced_index(i, disp, box_depth, self.nr_dims)
                box.neighbors.append(neighbor_idx)
                box.neighbor_displacements.append(disp)
                box.neighbor_disp_dict[disp.tobytes()] = neighbor_idx


    def calculate_grid_spacings(self):
        # Calculates and returns a list of grid spacings in all boxes.
        # At the moment just returns 2 for all but the middle box, which has spacing 1.
        a_list = 2*np.ones(self.nr_boxes, dtype=int)
        a_list[self.nr_boxes//2] = 1
        return a_list


    def get_nearby_points(self, idx, D, return_disp=False):
        """ Returns the index and displacement of points within D*a_local finegrid
        steps, where a_local is the largest local steplength."""
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.Box_list[box_nr]
        a_local = box.a_local
        disp = get_combinations(-D, D, self.nr_dims)
        coord_local = box.coords_local[idx-box.start_idx_global]
        coord_global = self.coords[idx]
        coords_local = coord_local + disp

        ### CASE 1: Point is entirely inside one box.
        if not self.IsCloseToEdge(idx, D, nr_dims=self.nr_dims):
            idxs = unravel((coords_local.T), box.N_local) + box.start_idx_global

        ### CASE 2: Point is within D of other box(es), but they all have the same spacing.
        elif not self.IsCloseToEdgeWithDifferentSpacing(idx, D, self.nr_dims):  # If nearby points crosses into one or more other boxes, but they have identical spacing.
            idxs = np.zeros((1+2*D)**self.nr_dims, dtype=int)-1
            neighbor_disp = (coords_local > box.N_local-1).astype(int) - (coords_local < 0).astype(int)

            for i in range(disp.shape[0]):
                if (neighbor_disp[i] == 0).all(): # If this point is in our box.
                    idxs[i] = unravel(coords_local[i], box.N_local) + box.start_idx_global
                else:
                    neighbor_box_nr = box.neighbor_disp_dict[neighbor_disp[i].tobytes()]
                    neighbor_box = self.Box_list[neighbor_box_nr]
                    neighbor_coord = coords_local[i].copy() - neighbor_disp[i]*box.N_local
                    neighbor_coord = (neighbor_coord*box.a_local)//neighbor_box.a_local
                    idxs[i] = unravel(neighbor_coord, neighbor_box.N_local) + neighbor_box.start_idx_global

        ### CASE 3: Point is within D of other box(es), and they have different spacing.
        else:
            idxs = []
            neighbor_disp = (coords_local > box.N_local-1).astype(int) - (coords_local < 0).astype(int)
            unique_neighbor_disp = np.unique(neighbor_disp, axis=0)

            # FINDING BIGGEST LOCAL a:
            a_local = box.a_local
            for i in range(len(unique_neighbor_disp)):
                neighbor_box_nr = box.neighbor_disp_dict[unique_neighbor_disp[i].tobytes()]
                if self.Box_list[neighbor_box_nr].a_local > a_local:  # If we find another box with higher a, that is the new local a.
                    a_local = self.Box_list[neighbor_box_nr].a_local

            disp = (disp*a_local)//box.a_local
            coords_local = coord_local + disp
            neighbor_disp = (coords_local > box.N_local-1).astype(int) - (coords_local < 0).astype(int)
            unique_neighbor_disp = np.unique(neighbor_disp, axis=0)

            # SEARCHING FOR ALL POINTS WITHIN SET GRID SPACING
            searchcube = np.repeat([[-1,1]], self.nr_dims, axis=0)*a_local*D + coord_local[:,None]*box.a_local
            idxs = []
            for i in range(len(unique_neighbor_disp)):
                neighbor_box_nr = box.neighbor_disp_dict[unique_neighbor_disp[i].tobytes()]
                current_box = self.Box_list[neighbor_box_nr]
                current_searchcube = searchcube - unique_neighbor_disp[i][:,None]*box.N_fine
                current_searchcube.clip(min=0, max=current_box.N_fine-1, out=current_searchcube)
                idxs_in_box = current_box.get_points_in_cube(current_searchcube)
                idxs.extend(idxs_in_box)

            # FINALLY, REMOVE POINTS THAT ARE NOT MIRROR-SYMETRIC ABOUT THE CENTER COORD.
            # Obs: Now dealing only in fine-grid coords. This is very easy to fuck up.
            # if only_symetric:
            cut_idxs = np.ones(len(idxs), dtype=bool)
            for i in range(len(idxs)):
                current_coord = self.coords[idxs[i]]
                mirror_coord = coord_global - (current_coord - coord_global)
                neighbor_disp = (mirror_coord-box.corner_coord > box.N_fine-1).astype(int) - (mirror_coord-box.corner_coord < 0).astype(int)
                neighbor_box_nr = box.neighbor_disp_dict[neighbor_disp.tobytes()]
                neighbor_box = self.Box_list[neighbor_box_nr]
                if (mirror_coord%neighbor_box.a_local != 0).any():
                    cut_idxs[i] = False
            idxs = np.array(idxs)[cut_idxs]
        return np.array(idxs)


        return idxs
        

    def IsCloseToEdge(self, idx, D, nr_dims):
        # Returns True if idx is within D steps from a neighboring box, and False otherwise.
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.Box_list[box_nr]
        coords = np.array(ravel(idx - box.start_idx_global, box.N_local, self.nr_dims))  # Calculate the "internal" , where 0,0,0 corresponds to box corner.
        disp = get_combinations(-1, 1, self.nr_dims)*D
        neighbor_disp = (coords + disp > box.N_local-1).astype(int) - (coords + disp < 0).astype(int)
        return len(np.unique(neighbor_disp, axis=0)) > 1


    def IsCloseToEdgeWithDifferentSpacing(self, idx, D, nr_dims):
        # Returns True if idx is withing D steps from a neighboring box with different spacing than its own, and False otherwise.
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.Box_list[box_nr]
        coords = np.array(ravel(idx - box.start_idx_global, box.N_local, self.nr_dims))  # Calculate the "internal" i,j,k, where 0,0,0 corresponds to box corner.
        disp = get_combinations(-1, 1, self.nr_dims)*D
        neighbor_disp = (coords + disp > box.N_local-1).astype(int) - (coords + disp < 0).astype(int)
        neighbor_disp = np.unique(neighbor_disp, axis=0)
        for x in neighbor_disp:
            if (x != 0).any():
                neighbor_nr = box.neighbor_disp_dict[x.tobytes()]
                if self.a_list[neighbor_nr] != box.a_local:
                    return True
        return False

    def IsCloseToEdgeWithHigherSpacing(self, idx, D, nr_dims):
        # Returns True if idx is withing D steps from a neighboring box with higher spacing than its own, and False otherwise.
        box_nr = self.get_box_nr_from_idx[idx]
        box = self.Box_list[box_nr]
        coords = np.array(ravel(idx - box.start_idx_global, box.N_local, self.nr_dims))  # Calculate the "internal" i,j,k, where 0,0,0 corresponds to box corner.
        disp = get_combinations(-1, 1, self.nr_dims)*D
        neighbor_disp = (coords + disp > box.N_local-1).astype(int) - (coords + disp < 0).astype(int)
        neighbor_disp = np.unique(neighbor_disp, axis=0)
        for x in neighbor_disp:
            if (x != 0).any():
                neighbor_nr = box.neighbor_disp_dict[x.tobytes()]
                if self.a_list[neighbor_nr] > box.a_local:
                    return True
        return False
