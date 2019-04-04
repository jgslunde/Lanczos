""" File containing random usefull functions that doesn't really belong anywhere in particular."""

def unravel_ijk(i, j, k, N):
    # Given i,j,k indexes, returns the unraveled index idx.
    return i + j*N + k*N**2


def ravel_idx(idx, N):
    # Given the unraveled index idx, gives the corresponding raveled i,j,k indexes.
    i = idx%N
    j = (idx//N)%N
    k = idx//N**2
    return i, j, k