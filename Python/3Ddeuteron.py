import time
import sys
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from Lanczos import Lanczos

def plot_potential(x, y, V):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surface = ax.plot_surface(X, Y, V, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
    plt.colorbar(surface)
    plt.show()

def unravel_xyz(x, y, z):
    # Given x, y, z indexes, gives the unraveled index i.
    return x + y*N + z*N**2

def ravel_i(i):
    # Given the unraveled index i, gives the corresponding x,y,z indexes.
    x = i%N
    y = (i//N)%N
    z = i//N**2
    return (x, y, z)

def ravel_array(oneD_array):
    threeD_array = np.zeros((N,N,N))
    for i in range(N**3):
        threeD_array[ravel_i(i)] = oneD_array[i]
    return threeD_array


def plot_radial_wavefunc(oneD_array):
    # Given a unraveled 1D array, plots the radial parts of the wavefunction.
    threeD_array = ravel_array(oneD_array)
    plt.plot(threeD_array[N//2:, :,:])







################################################################################
################################################################################

# Potential V matrix setup
def potential(x, y, z):
    # Returns potential at location (x, y, z) for a box of size L with center in L/2 (all dim.), and deteron potential.
    r = np.sqrt(x**2 + y**2 + z**2)
    # Deuteron setup
    eWell = 54.531
    eWells = 65.4823128982115
    eCores = 40.0*eWell
    rCore = 1.0/4
    rWell = 17.0/10
    fPow = 4.0
    return eCores*np.exp(-(r/rCore)**fPow) - eWells*np.exp(-(r/rWell)**fPow)

N = 40
n = 160
L = 12 # Length of system in fm.
dx = float(L)/N

# Derivative T matrix setup
hc = 197.327 # MeV_fm
rest_energy = 469.4592 # MeV / c^2
T_factor = hc**2/(2*rest_energy) * 1/dx**2


from Hamiltonian import Hamiltonian
system = Hamiltonian(N, L, potential, T_factor)   
system.create_sparse_T()
system.create_sparse_V()
T_sparse = system.T_sparse
V_sparse = system.V_sparse
H = (-T_sparse + V_sparse)

# Hamiltonian 
#H = (-T + V)
print("H MATRIX:")
print(H)
# print(np.array_str(H, precision=2, suppress_small=True))

print("V MATRIX:")
print(V_sparse)
# print(np.array_str(V, precision=2, suppress_small=True))

print("T MATRIX:")
print(T_sparse)
# print(np.array_str(T, precision=2, suppress_small=True))


# Running Lanczos
TEST = Lanczos(H)
TEST.execute_Lanczos(n, use_cuda=True)
l_L, v_L = TEST.H_eigvals, TEST.H_eigvecs
TEST.print_good_eigs()

# Comparing to analytical results.
# TEST.compare_eigs()

# Plotting groundstate
# l, v = scipy.sparse.linalg.eigsh(H)
# numpy_groundstate = ravel_array(v[:,0])
#plt.plot(numpy_groundstate[:, N//2, N//2])
#plt.show()