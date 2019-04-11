""" File containing random usefull functions that doesn't really belong anywhere in particular."""
import numpy as np


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

