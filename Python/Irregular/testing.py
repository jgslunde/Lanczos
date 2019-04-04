""" Tests of anything, really."""
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange
from IrrGrid import IrrGrid


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














if __name__ == "__main__":
    Test_Plot_IsCloseToEdge()