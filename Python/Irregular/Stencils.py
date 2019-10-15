import numpy as np
from tools2 import get_relative_positions

class Stencils:
    def __init__(self, lat):
        self.lat = lat
        self.hash_table = {}

    def calc_hash(self, idx):
        # Calculates the unique has of the lattice point idx.
        lat = self.lat
        if not lat.IsCloseToEdgeWithHigherSpacing(idx, 1, nr_dims=lat.nr_dims):
            return 0  # If we're not within 1 step length of any box with different spacing, the stencil will just be the 27-point stencil.
        
        else:
            coord = lat.coords[idx]
            box_nr = lat.get_box_nr_from_idx[idx]
            box = lat.Box_list[box_nr]
            coord_local = box.coords_local[idx-box.start_idx_global]
            
            distance_up = box.N_local - coord_local
            distance_down = coord_local+1
            on_grid = coord_local%2 == 0
            distance = np.where(distance_up < distance_down, distance_up, -distance_down)
            print(distance_up, distance_down, distance, on_grid)
            distance = np.where(np.abs(distance) < 5, distance, np.zeros(len(distance)))

            hash_array = (2*distance + 10) + on_grid
            hash = 0
            for i in range(len(hash_array)):
                hash += int(hash_array[i]*(20**i))

            hash = lat.get_nearby_points(idx, 2, lat.nr_dims)
            hash = hash.tobytes()

            return hash


    def get_stencil(self, idx):
        hash = self.calc_hash(idx)
        if hash in self.hash_table:
            return self.hash_table[hash]
        else:
            box = self.lat.Box_list[self.lat.get_box_nr_from_idx[idx]]
            coord = self.lat.coords[idx]
            if not self.lat.IsCloseToEdgeWithHigherSpacing(idx, 1, self.lat.nr_dims):
                search_dist = 1
            else:
                search_dist = 2
            neighbor_idxs = self.lat.get_nearby_points(idx, search_dist)
            neighbor_coords = self.lat.coords[neighbor_idxs]
            neighbor_disp = get_relative_positions(coord, neighbor_coords, self.lat.N, self.lat.nr_dims)/box.a_local

            self.hash_table[hash] = neighbor_disp
            return neighbor_disp







def test_Stencils():
    from Lattice import Lattice
    N = 30
    L = 1
    nr_dims = 2
    box_depth = 3
    lat = Lattice(N, L, nr_dims)
    lat.setup_boxes(box_depth)
    stencils = Stencils(lat)
    for idx in range(lat.nr_points):
        s = stencils.get_stencil(idx)

    for idx in range(lat.nr_points):
        if not lat.IsCloseToEdgeWithHigherSpacing(idx, 1, nr_dims):
            nr_points = len(lat.get_nearby_points(idx, 1))
        else:
            nr_points = len(lat.get_nearby_points(idx, 2))
        if len(s) != nr_points:
            print(len(s), "\n", nr_points, s)

        

    #print(stencils.hash_table)
    #print(len(stencils.hash_table))

if __name__ == "__main__":
    test_Stencils()