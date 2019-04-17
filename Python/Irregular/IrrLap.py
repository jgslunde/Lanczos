import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def HashListKen(points, shift = 256):
    # Hasing an unordered [n, d] list of points.
    h = 123
    nr_points, d = points.shape

    for i in range(nr_points):
        for a in range(d):
            h += points[i,a] << a
        h += (h >> 3)
    h &= shift - 1
    return h


hashtable = {}
def HashList(points, bits_per_point = 10):  # bit-shift of 5 means that values up to 2^5 = 32 cause no hash-colisions.
    # Hasing an unordered [n, 3] list of points.
    nr_points, d = points.shape
    h = 0
    for i in range(nr_points):
        h += (points[i,0] + points[i,1]*32 + points[i,2]*32**2)*32**(3*i)
    # h_arr = np.zeros(nr_points, dtype=np.int32)

    # for i in range(nr_points):
    #     h_arr[i] += points[i,0] + (points[i,1]<<bits_per_point) + (points[i,2]<<bits_per_point*2)

    # h = 0
    # for i in range(nr_points):
    #     h += (h_arr[i]<<(bits_per_point*i))
    return h

def Laplacian(points):
    """
    INPUT: points - [nr_points, 3] array of grid-points by their relative offset from "center point"
                    (the point we are calculating the Laplacian of) in units of the fine-grid spacing.
    RETURNS: weights [nr_points] array of the caluclated weights for the Laplacian."""

    hashval = HashList(points)

    if hashval in hashtable:
        weights = hashtable[hashval]

    else:
        nr_points = points.shape[0]
        weights = np.zeros(nr_points)

        w = np.zeros(nr_points)  # Weighing of each point when calculating Laplacian, determined by distance from center point.
                                # Not to be confused with "weights", which is the final weights in the Laplacian.
        # Setting weights by distance from center point:
        for i in range(nr_points):
            r = points[i,0]**2 + points[i,1]**2 + points[i,2]**2
            if r == 0:
                print("Warning, r=0, meaning origin was included. Weight for origins are NOT calculated, and will return 0.")
            else:
                w[i] = 1/r**2  # weigth is 1 over distance from 0 squared.

        dim = 9; d = 3  # = d + d*(d+1)/2, d=3
        M = np.zeros((dim, dim))

        row = 0; col = 0
        for g in range(3):
            for a in range(3):
                for i in range(nr_points):
                    M[row, col] += w[i]*points[i,a]*points[i,g]
                col += 1
            row += 1; col = 0

        row = 0; col = 3
        for g in range(3):
            for a in range(3):
                for b in range(a, 3):
                    for i in range(nr_points):
                        M[row, col] += w[i]*points[i,a]*points[i,b]*points[i,g]
                    col += 1
            row += 1; col = 3

        row = 3; col = 0
        for g in range(3):
            for c in range(g, 3):
                for a in range(3):
                    for i in range(nr_points):
                        M[row, col] += w[i]*points[i,a]*points[i,g]*points[i,c]
                    col += 1
                row += 1; col = 0

        row = 3; col = 3
        for g in range(3):
            for c in range(g, 3):
                for a in range(3):
                    for b in range(a, 3):
                        for i in range(nr_points):
                            M[row, col] += w[i]*points[i,a]*points[i,b]*points[i,g]*points[i,c]
                        col += 1
                row += 1; col = 3

        M_inv = np.linalg.inv(M)

        mit = np.zeros(dim)

        row = d
        for a in range(d):
            for b in range(a, d):
                if a == b:
                    for i in range(dim):
                        mit[i] += M_inv[row, i]
                row += 1

        pos = 0
        for a in range(d):
            for i in range(nr_points):
                weights[i] += mit[pos]*w[i]*points[i,a]
            pos += 1

        for a in range(d):
            for b in range(a, d):
                for i in range(nr_points):
                    weights[i] += mit[pos]*w[i]*points[i,a]*points[i,b]
                pos += 1

        hashtable[hashval] = weights
    return weights





if __name__ == "__main__":
    # points = np.array([[0,0,0],
    # [2,0,0],
    # [0,2,0],
    # [0,0,2],
    # [-2,0,0],
    # [0,-2,0],
    # [0,0,-2],
    # [1,1,0],
    # [1,0,1],
    # [0,1,1],
    # [-1,-1,0],
    # [-1,0,-1],
    # [0,-1,-1],
    # [-1,1,0],
    # [1,-1,0],
    # [-1,0,1],
    # [1,0,-1],
    # [0,-1,1],
    # [0,1,-1]])

    
    points_27 = np.array([[i,j,k] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2)])


    test = Laplacian(points_27)


    N = 60
    L = 25

    dx = float(L)/N
    hc = 197.327 # MeV_fm
    rest_energy = 469.4592 # MeV / c^2
    T_factor = hc**2/(2*rest_energy) * 1/dx**2
    print(T_factor)

    for i in range(27):
        print(points_27[i], test[i])

    """
    hashKen = []
    hashJonas = []
    pointsList = []
    conflicts = 0
    for i in range(int(1e4)):
        points = np.random.randint(-3, 4, (27, 3))
        hK = HashListKen(points)
        hJ = HashList(points)
        if hK in hashKen:
            match = np.argwhere(hK == hashKen)
            #print("\nCONFLICT in Ken, HASH = ", hK)
            #print(" POINTS 1 = \n", pointsList[match[0,0]])
            #print(" POINTS 2 = \n", points)
            conflicts += 1

        if hJ in hashJonas:
            print("\nCONFLIC in Jonas")
        
        hashKen.append(hK)
        hashJonas.append(hJ)
        pointsList.append(points)
        
    print(conflicts)



    import time
    t0 = time.time()
    HashListKen(points)
    print(f"{time.time() - t0:.8f}")
    t0 = time.time()
    HashList(points)
    print(f"{time.time() - t0:.8f}")
    """