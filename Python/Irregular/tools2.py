import numpy as np
from itertools import product

def unravel(coords, N):
    # Unravels indexable variable "coords" into 1D indexes.
    result = 0
    for i in range(len(coords)):
        result += coords[i]*N**i
    return result


def unravel2(N, x1, x2, x3, x4=None, x5=None, x6=None):
    # Alternative syntax for unraveling.
    if x4==None:
        return x1 + x2*N + x3*N**2
    else:
        return x1 + x2*N + x3*N**2 + x4*N**3 + x5*N**4 + x6*N**5


def ravel(idx, N, nr_dims):
    # Ravels an index into a position vector in a chosen dimension.
    if nr_dims == 3:
        x1 = idx%N
        x2 = (idx//N)%N
        x3 = idx//N**2
        return x1, x2, x3
    elif nr_dims == 6:
        x1 = idx%N
        x2 = (idx//N)%N
        x3 = idx//N**2
        x4 = (idx//N**2)%N
        x5 = idx//N**3
        x6 = (idx//N**3)%N
        return x1, x2, x3, x4, x5, x6
    elif nr_dims == 2:
        x1 = idx%N
        x2 = (idx//N)%N
        return x1, x2
    else:
        raise NotImplementedError("Dims other that 2, 3, and 6 not implemented.")


def get_combinations(x_start, x_stop, D):
    # Returns a 2D array of all unique ways of filling a D long array with
    # values in the (inclusive) integer interval [x_start, x_stop].
    # Example: x_start = 0, x_stop = 2, D = 3. Returns:
    # [[0,0,0], [1,0,0], [2,0,0,], [0,1,0], [1,1,0], ... , [1,2,2], [2,2,2]]
    return np.array(list(product(range(x_start, x_stop+1), repeat=D)))[:,::-1]


def get_displaced_index(idx, disp, N, nr_dims):
    # Returns the index of the point with displacement "disp" from the point
    # with index "idx", using wraparound in a box with shape [0, N-1] in each dim.
    # disp should be a legnth "nr_dims" vector of displacements along each axis.
    coords = np.array(ravel(idx, N, nr_dims))  # Ravel idx into coords.
    for i in range(len(coords)):
        coords[i] += disp[i]  # Add displacement.
        if coords[i] < 0:  # Employ wraparound if outsize bounds of [0, N-1].
            coords[i] = N + coords[i]
        elif coords[i] >= N:
            coords[i] = coords[i] - N
    return unravel(coords, N)


def get_displaced_coord(coord, disp, N, nr_dims):
    # In a nr_dims-dimensional box of size N in each dim, return the coordinate of the 
    # displacement "disp" from "coord", using wraparound on each axis.
    new_coord = coord + disp
    new_coord = np.where(new_coord < 0, N + new_coord, new_coord)
    new_coord = np.where(new_coord >= N, new_coord - N, new_coord)
    return new_coord


def get_relative_positions(center_pos, pos_array, N, nr_dims):
    # Return the relative position of pos_array compared to center_pos on a grid which wraps at N in each dimension.
    delta_pos = np.empty((pos_array.shape[0], nr_dims), dtype=int)
    for i in range(nr_dims):  # Distances in each dimension treated seperatly.
        direct = pos_array[:,i] - center_pos[i]
        wraparound = np.where(direct > 0, direct-N, N+direct)
        delta_pos[:,i] = np.where(np.abs(direct) - np.abs(wraparound) < 0, direct, wraparound)   
    return delta_pos