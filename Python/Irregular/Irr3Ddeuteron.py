import time
import sys
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from IrrLanczos import IrrLanczos
from IrrHamiltonian import Hamiltonian
from IrrGrid import IrrGrid
from Potentials import Deuterium3DPotential

N = 48
L = 25
box_depth = 1

dx = float(L)/N

hc = 197.327 # MeV_fm
rest_energy = 469.4592 # MeV / c^2
T_factor = hc**2/(2*rest_energy) * 1/dx**2

Grid = IrrGrid(N, L)
Grid.SetupBoxes(box_depth=box_depth)
Ham = Hamiltonian(Grid, Deuterium3DPotential, T_factor)
Ham.MakeSparseH()

print("H:")
print(Ham.H_sparse)

print("V:")
print(Ham.V_sparse)

print("T:")
print(Ham.T_sparse)

L = IrrLanczos(Ham.H_sparse)
L.execute_Lanczos(100)
L.get_H_eigs()
L.print_good_eigs()