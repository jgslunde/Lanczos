""" Tests of anything, really."""
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange
from IrrGrid import IrrGrid
from symetry import FindMirrorSymetricPoints


def Test_Plot_IsCloseToEdge():
    N = 50
    L = 25
    grid = IrrGrid(N, L)
    grid.SetupBoxes(box_depth=5)
    print(grid.nr_points)
    asdf = np.zeros((N,N,N))

    for i in trange(N):
        for j in range(N):
            for k in range(N):
                idx = i+j*N+k*N**2
                if grid.IsCloseToEdge(idx, 1) and grid.point_coords[idx,2] == 13:
                    plt.scatter(*grid.point_coords[idx,:2], c="r", s=10)
                elif grid.point_coords[idx,2] == 13:
                    plt.scatter(*grid.point_coords[idx,:2], c="b", s=10)
    plt.show()




def Test_Plot_GetNearbyPoints():
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
    Test_Plot_GetNearbyPoints()
    Test_Plot_IsCloseToEdge()