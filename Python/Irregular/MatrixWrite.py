### HOW TO USE - Execute the program in python, with the optional arguments (with their defaults as examples):
### -d 3    :Number of dimensions in system. Currently only 3 supported.
### -L 25    :Length of box in each dimension, in fm.
### -N 30    :Number of fine-grid points in every dimension. Note that since the grid is irregular, the total number of points will be much smaller.
### -p "Deuteron"    :Potential type. Currently only "Deuteron" supported.
### Full command line example:
### $python3 MatrixWrite.py -d 3 -L 25 -N 30 -p "Deuteron"


import numpy as np
from scipy import sparse
from IrrGrid import IrrGrid
from Potentials import Deuterium3DPotential
from IrrHamiltonian import Hamiltonian


def MatrixWrite(d, L, N, p):
    if p == "Deuteron":
        potential = Deuterium3DPotential
    else:
        raise NotImplementedError("Other Potentials not implemented yet.")

    if d != 3:
        raise NotImplementedError("Dimensions other than 3 not implemented yet.")

    box_depth = 3
    dx = float(L)/N
    hc = 197.327 # MeV_fm
    rest_energy = 469.4592 # MeV / c^2
    T_factor = hc**2/(2*rest_energy) * 1/dx**2 * 2
    Grid = IrrGrid(N, L, overwrite_spacing=True)
    Grid.SetupBoxes(box_depth=box_depth)
    Ham = Hamiltonian(Grid, potential, T_factor, normal_eq=False)
    Ham.MakeSparseH()

    ### Wiring to file ###
    string = f"""numd = {d:d};
nrpoints = {Ham.H_sparse.count_nonzero():d};
box = {{{L:g}, {L:g}, {L:g}}};
potential = \"{p}\";
H = {{{{{N**3:d}, {N**3:d}}}, {{"""

    H = sparse.coo_matrix(Ham.H_sparse)
    row, col, data = H.row, H.col, H.data
    l = len(row)

    row = np.array(row, dtype=str)
    col = np.array(col, dtype=str)
    data2 = np.array([f"{data[i]:.17f}" for i in range(l)], dtype=str)
    data = data2; del data2

    row = np.array(row, dtype=object)
    col = np.array(col, dtype=object)
    data = np.array(data, dtype=object)

    temp1 = np.repeat("{", l)
    temp2 = np.repeat(", ", l)
    temp3 = np.repeat("},\n", l)
    string += "".join(temp1 + row + temp2 + col + temp2 + data + temp3)
    string += "}};"
    with open(f"matrix_d={d}_N={N}_L={L}_p={p}.dat", "w") as outfile:
        outfile.write(string)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=3, help="Number of dimensions.")
    parser.add_argument("-L", type=float, default=25, help="Side lengths of simulation box, in fm.")
    parser.add_argument("-N", type=int, default=30, help="Number of fine-grid points in each dimension.")
    parser.add_argument("-p", type=str, default="Deuteron", help="Potential name.")

    args = parser.parse_args()
    d, L, N, p = args.d, args.L, args.N, args.p
    MatrixWrite(d, L, N, p)