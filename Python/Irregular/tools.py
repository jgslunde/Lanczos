""" File containing random usefull functions that doesn't really belong anywhere in particular."""
import numpy as np
from itertools import product


def unravel_ijk(i, j, k, N):
    # Given i,j,k indexes, returns the unraveled index idx.
    return i + j*N + k*N**2


def ravel_idx(idx, N):
    # Given the unraveled index idx, gives the corresponding raveled i,j,k indexes.
    i = idx%N
    j = (idx//N)%N
    k = idx//N**2
    return i, j, k


def get_relative_positions(center_pos, pos_array, N):
    # Return the relative position of pos_array compared to center_pos on a grid which wraps at N in each dimension.
    delta_pos = np.empty((pos_array.shape[0], 3), dtype=int)
    for i in range(3):  # Distances in each dimension treated seperatly.
        direct = pos_array[:,i] - center_pos[i]
        wraparound = np.where(direct > 0, direct-N, N+direct)
        delta_pos[:,i] = np.where(np.abs(direct) - np.abs(wraparound) < 0, direct, wraparound)   
    return delta_pos



hashdict = {}
def get_symetric_displacement_stencil(D, d):
    # Returns a [(2*D+1)^d, d] array with all possible displacements around the origin, up to "D" in each direction.
    hash = D+d*128  # Creating a unique hash for the inputs.
    if hash in hashdict:
        return hashdict[hash]
    else:
        disp = np.array(list(product(range(-D, D+1), repeat=d)))
        hashdict[hash] = disp
    return disp


hashdict2 = {}
def get_displacement_stencil(searchcube, d):
    hash = searchcube.tobytes()
    if hash in hashdict2:
        return hashdict2[hash]
    else:
        disp = np.array(list(product(searchcube, repeat=d)))
        hashdict2[hash] = disp
    return disp



# def get_displacement_stencil(D, on_grid, a, stretch_offgrid=True):
#     """INPUT:
#     D = Int. Depth of points in each direction.
#     on_grid = [dims] boolean array, representing if each dimension is on or off grid.
#     a = default spacing in each dimension.
#     stretch_offgrid = Option to include +1 depth in offgrid direction.
#     OUTPUT:
#     """

#     disp_options = []
#     for dim in range(3):
#         if on_grid[dim]:
#             disp_options.append(np.arange(-D, D+1, a))
#         else:
#             disp_options.append(np.concatenate((np.arange(-D, 0, a), np.arange(a, D+1, a)), axis=None))

#         disp = np.array(list(repeat(disp_options)))
#     return disp



if __name__ == "__main__":
    for i in range(int(1e6)):
        disp = get_symetric_displacement(2, 3)