import numpy as np
import matplotlib.pyplot as plt
from Lattice import Lattice



if __name__ == "__main__":
    nr_dims = 2
    N = 33
    L = 1
    box_depth = 3

    lat = Lattice(N, L, nr_dims=nr_dims)
    lat.setup_boxes(box_depth)
    hash_list = np.zeros(lat.nr_points, dtype=int)
    idx_list = np.zeros(400, dtype=int)

    # for idx0 in range(lat.nr_points):
    #     hash = lat.get_stencil(idx0)
    #     hash_list[idx0] = hash

    #     idx_list[hash] = idx0


    # plt.hist(idx_list)
    # plt.show()

    for idx in range(lat.nr_points):
        coord = lat.coords[idx]
        plt.scatter(coord[0], coord[1], c="navy")
        for i in range(box_depth+1):
            plt.axhline(i*N//box_depth - 0.5, c="y", ls="--")
            plt.axvline(i*N//box_depth - 0.5, c="y", ls="--")

        plt.axis("equal")
    plt.show()



    # start_idx = 198
    # print(lat.coords[start_idx])
    # idxs = lat.get_nearby_points(start_idx, 2)
    # for idx in idxs:
    #     coord = lat.coords[idx]
    #     plt.scatter(coord[0], coord[1], c="crimson")
    # plt.scatter(lat.coords[start_idx][0], lat.coords[start_idx][1], c="y")

    # plt.show()