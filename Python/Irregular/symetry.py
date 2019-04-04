import numpy as np
import matplotlib.pyplot as plt
import time
from sympy.utilities.iterables import multiset_permutations

def FindMirrorSymetricPoints(points, min_nr_points=9):
    #TODO switch out sympy multiset_permutations with a preloaded array, from iterables.producs.
    # Takes a [N,d] array of N points, in D dimensions.
    # Returns a [n] array of point indexes, the indexes being the n=<N points which have mirror-symetry over all axis.
    N = points.shape[0]
    d = points.shape[1]
    symetric_points = []
    for idx in range(N):  # Looping over every point to see if it's symetric.
        if idx not in symetric_points:
            mirror_point_idxs = []  # A list containting the indexes of the mirror-twins of this point, as these must also be symetric.
            point_is_symetric = True
            for dim in range(d):  # Looping over the numbers of dimensions we will "flip". (I.e. for dim=2, we will mirror the point in 2 dimensions).
                mirroring_temp = np.ones(d, dtype=int)  # We create a simple array with [1,1,...-1,-1], representing how many axis we will "flip".
                mirroring_temp[:dim+1] = -1
                for mirroring in multiset_permutations(mirroring_temp):  # We then loop over every possible (unique) permutation of this, representing a unique set of axis mirroring.
                    mirrored_point = points[idx].copy()*mirroring
                    found_symetry = False
                    for k in range(N):
                        if (mirrored_point == points[k]).all():
                            found_symetry = True
                            mirror_point_idxs.append(k)
                    if not found_symetry:  # If the mirror of the point does not exist in one of the mirrored axis, the point is not symetric.
                        point_is_symetric = False
            if point_is_symetric:
                symetric_points.append(idx)
                symetric_points.extend(mirror_point_idxs)  # The mirror-twin points must also be symtric, so we simply add them.

    if len(symetric_points) >= min_nr_points:
        return np.array(symetric_points)
    else:
        raise ValueError(f"Found only {len(symetric_points)} symetric points for stencil!")



if __name__ == "__main__":
    N = 8
    np.random.seed(54)
    points = np.array([[-4, 3], [-2, 3], [0, 3], [2, 3], [4, 3],\
                       [-4, 1], [-3, 1], [-2, 1], [-1,1], [0, 1], [2, 1], [4, 1],\
                       [-4, 0], [-3, 0], [-2, 0], [-1, 0], [0,0],\
                       [-4,-1], [-3,-1], [-2,-1], [-1,-1], [0,-1], [2,-1], [4,-1],\
                       [-4,-2], [-3,-2], [-2,-2], [-1,-2], [0,-2],\
                       [-4,-3], [-3,-3], [-2,-3], [-1,-3], [0,-3], [2,-3], [4,-3],\
                       [-4,-4], [-3,-4], [-2,-4], [-1,-4], [0,-4]])

    # points = np.unique(np.random.randint(-16, 17, (601, 2)), axis=0)
    t0 = time.time()
    sym_idxs = FindMirrorSymetricPoints(points)
    print(time.time()-t0)
    sym_points = points[sym_idxs]

    plt.scatter(*points.T, c="r")
    plt.scatter(*sym_points.T, c="b")
    plt.axhline(y=0, c="y", ls="--")
    plt.axvline(x=0, c="y", ls="--")
    plt.show()

    plt.scatter(*sym_points.T, c="b")
    plt.axhline(y=0, c="y", ls="--")
    plt.axvline(x=0, c="y", ls="--")
    plt.show()