import numpy as np
import matplotlib.pyplot as plt
from Lanczos import Lanczos
import scipy.sparse

N = 1001
n = 1001
L = 25 # Length of system in fm.
dx = float(L)/N

# Deuteron setup
eWell = 54.531
eWells = 65.4823128982115
eCores = 40.0*eWell
rCore = 1.0/4
rWell = 17.0/10
fPow = 4.0

# Potential V matrix setup
r = np.linspace(0, L, N)
V_array = eCores*np.exp(-(r/rCore)**fPow) - eWells*np.exp(-(r/rWell)**fPow)
# V = np.diag(V_array)
row_ind = []; col_ind = []; data = []
for i in range(N-1):
    row_ind.append(i); col_ind.append(i); data.append(V_array[i])
V_sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N,N))


# Derivative T matrix setup
hc = 197.327 # MeV_fm
rest_energy = 469.4592 # MeV / c^2
T_factor = hc**2/(2*rest_energy) * 1/dx**2

row_ind = []; col_ind = []; data = []
row_ind.append(0); col_ind.append(0); data.append(-1*T_factor)
row_ind.append(0); col_ind.append(1); data.append(1*T_factor)
row_ind.append(N-1); col_ind.append(N-2); data.append(1*T_factor)
row_ind.append(N-1); col_ind.append(N-1); data.append(-1*T_factor)
for i in range(1, N-1):
    row_ind.append(i); col_ind.append(i-1); data.append(1*T_factor)
    row_ind.append(i); col_ind.append(i); data.append(-2*T_factor)
    row_ind.append(i); col_ind.append(i+1); data.append(1*T_factor)
T_sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N,N))


# T = np.zeros((N,N))
# T[0, :2] = [-1*T_factor, T_factor]
# T[-1, -2:] = [T_factor, -1*T_factor]
# for i in range(1, N-1):
#     T[i, i-1:i+2] = [T_factor, -2*T_factor, T_factor]

# Hamiltonian
# H = (-T + V)
H = (-T_sparse + V_sparse)

print("H MATRIX:")
print(H)
#print(np.array_str(H, precision=2, suppress_small=True))

print("V MATRIX:")
print(V_sparse)
#print(np.array_str(V, precision=2, suppress_small=True))

print("T MATRIX:")
print(T_sparse)
#print(np.array_str(T, precision=2, suppress_small=True))


# Running Lanczos
# l, v = np.linalg.eigh(H)
TEST = Lanczos(H)
TEST.execute_Lanczos(n)
l_L, v_L = TEST.H_eigvals, TEST.H_eigvecs

# Comparing to analytical results.
TEST.compare_eigs()

"""
# Plotting best fits against analytical results.
v_sorted, vL_sorted, l_sorted, lL_sorted = Lanczos.get_matched_eigs(v, v_L, l, l_L)
for i in range(5):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    color2 = 'tab:green'
    ax1.set_xlabel('Radius [fm]')
    ax1.set_ylabel('Wavefunc', color=color)
    ax1.plot(r, v_sorted[:,i], color=color)
    ax1.plot(r, vL_sorted[:,i], color=color2, ls="--")
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Potential [MeV]', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(r, V_array, color=color)
    plt.title(f"{lL_sorted[i]:.2f} --- {l_sorted[i]:.2f} --- {np.dot(vL_sorted[:,i], v_sorted[:,i])**2:.4f}")
    plt.show()

"""