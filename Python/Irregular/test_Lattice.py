import numpy as np
from Lattice import Lattice


def teststuff():
    for dim in [2, 3, 6]:
        N = 30
        L = 1
        box_depth = 3
        lat = Lattice(N, L, nr_dims=dim)
        lat.setup_boxes(box_depth)

        assert lat.nr_boxes == len(lat.Box_list), f"Number of boxes doesn't correspond to length of Box_list, dim={dim}."
        assert len(lat.Box_list) == box_depth**dim, f"Number of boxes doesn't corresponds to length of Box_list, dim={dim}"
        for i in range(box_depth**dim):
            box = lat.Box_list[i]
            if (box.corner_coord == np.ones(dim)*(box_depth-1)//2).all():
                assert box.a_local == 1, f"Central box doesn't have spacing 1, dim={dim}"
            else:
                assert box.a_local == 2, f"Non-central box doesn't have spacing 2, dim={dim}"
        nr_points = np.sum((N//(box_depth*lat.a_list))**dim, dtype=int)
        assert lat.nr_points == nr_points, f"Number of points {nr_points}, doesn't correspond to reported number of points, {lat.nr_points}, dim={dim}."


if __name__ == "__main__":
    # teststuff()


    nr_dims = 2
    N = 33
    L = 1
    box_depth = 3

    lat = Lattice(N, L, nr_dims=nr_dims)
    lat.setup_boxes(box_depth)