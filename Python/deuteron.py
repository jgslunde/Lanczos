import numpy as np
import matplotlib.pyplot as plt
from Lanczos import Lanczos

eWell = 54.531
eWells = 65.4823128982115
eCores = 40.0*eWell
rCore = 1.0/4
rWell = 17.0/10
fPow = 4.0

N = 1001
n = 101
r = np.linspace(0, 2.5, N)
V = eCores*np.exp(-(r/rCore)**fPow) - eWells*np.exp(-(r/rWell)**fPow)

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
l_L, v_L = TEST.H_eigvals, TEST.H_eigvecs

TEST.compare_eigs(minimize="vec")



v_sorted, vL_sorted, l_sorted, lL_sorted = Lanczos.get_matched_eigs(v, v_L, l, l_L)

for i in range(5):
    plt.plot(r, v_sorted[:,i], c="k")
    plt.plot(r, vL_sorted[:,i], c="r", ls="--")
    plt.title(f"{lL_sorted[i]}, {l_sorted[i]}, {np.dot(vL_sorted[:,i], v_sorted[:,i])**2}")
    plt.show()


#idx = (np.dot(v_L, v)**2 ).argmax()


# print(l)
# plt.plot(r, V)
# plt.ylim(-80,100)
# plt.show()
# plt.plot(r, v[:,0])
# plt.show()
