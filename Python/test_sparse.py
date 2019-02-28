from scipy import sparse
import numpy as np
import time
import sys
N = int(4e3)
H_sparse = sparse.rand(N, N, 0.001, format="csc")
H_dense = H_sparse.toarray()

x = np.linspace(0, 10, N)

t0 = time.time()
for i in range(N):
    x = H_sparse * x
t1 = time.time()
print(f"Sparse time {N} matrix-vector multiplications = {t1-t0:.4f}")

t0 = time.time()
for i in range(N):
    x = np.dot(H_dense, x)
t1 = time.time()
print(f"Dense time {N} matrix-vector multiplications = {t1-t0:.4f}")