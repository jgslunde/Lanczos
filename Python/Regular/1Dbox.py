import numpy as np
import matplotlib.pyplot as plt
from Lanczos import Lanczos

N = 500
n = 50

V = np.zeros(N)
V[N//4 : (3*N)//4] = -10
#for i in range(N):
#    V[i] = -500 + 0.1*abs(i - N//2)**2

H = np.zeros((N,N))

H[0,0] = 2 + V[0]
H[0,1] = -1
H[-1,-1] = 2 + V[-1]
H[-1,-2] = -1
for i in range(1, N-1):
    H[i,i-1] = -1
    H[i,i+1] = -1
    H[i,i] = 2 + V[i]


l, v = np.linalg.eigh(H)

TEST = Lanczos(H)
TEST.execute_Lanczos(n)
TEST.get_H_eigs()
l_L, v_L = TEST.H_eigvals, TEST.H_eigvecs

TEST.compare_eigs(minimize="val")


for i in range(3):
    plt.plot(v[:,i], c = "b", label="Analytical eigvec.")
    plt.plot(v_L[:,i], ls="--", c="r", label="Lanczos Eigvec")
    plt.ylim(-0.2, 0.2)
    plt.legend()
    plt.show()
