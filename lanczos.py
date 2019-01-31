import numpy as np

def get_sorted_eigs(A):
    """
    INPUT:  A = (N,N) matrix.
 
    RETURN: eigvals = (N) array of sorted (high to low) eigenvalues of A.
            eigvecs = (N,N) array of sorted eigenvalues, such that
                       eigvecs[:,i] corresponds to eigvals[i].
    """
    eigvals, eigvecs = np.linalg.eig(A)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return eigvals, eigvecs


def print_vec(x):
    print("--------------")
    for i in range(len(x)):
        print("[ %12.4f ]" % x[i])
    print("--------------")


def print_2vec(x, y, n):
    print(np.linalg.norm(x), np.linalg.norm(y))
    print("--------------")
    for i in range(n):
        print("[ %12.4f ], [ %12.4f ] DIFF = %.4f" % (x[i], y[i], abs(y[i]-x[i])))
    print("--------------")    


def test_is_Hermitian(A):
    """Tests if A is Hermitian."""
    assert (np.matrix(H) == np.matrix(H).H).all(),\
    "A IS NOT HERMITIAN!"


def test_is_normalized(V, tol=0.001, no_assert=False):
    """ INPUT: V = (M,m) matrix. Tests that it's columns V[:,i] are normal.
               no_assert = Setting this to True will NOT run a test, and instead
                            return the inner product of the least normal vector.
    """
    least_normalized = 1
    for i in range(np.shape(V)[1]):
        normalization = np.linalg.norm(V[:,i])
        if abs(normalization - 1) > (least_normalized - 1):
            least_normalized = normalization
    if no_assert:
        return least_normalized
    else:
        assert abs(least_normalized-1) < tol,\
        "VECTOR HAS NORM %.4f. IS NOT NORMALIZED." % least_normalized


def test_is_orthogonal(V, tol=0.01, no_assert=False):
    """Tests whether the column vectors of V[:,i] are all orthogonal."""
    M = np.shape(V)[1]
    test_matrix = np.abs(np.matrix(V).T * np.matrix(V) - np.eye(M))
    max_error_idx = np.unravel_index(np.argmax(test_matrix, axis=None), test_matrix.shape)
    max_error = test_matrix[max_error_idx]
    if no_assert:
        return max_error
    else:
        assert max_error < tol,\
        "VECTORS %d AND %d NOT ORTHOGONAL! INNER PRODUCT %.4f" %\
        (max_error_idx[0], max_error_idx[1], max_error)


def test_Heff_and_V(H, H_eff, V, no_assert=False):
    """ Woopwoop: OUT OF ORDER. CHeck later. """
    N = np.shape(H)[0]
    error = np.zeros(N)
    error[0] = np.sum(np.abs(np.dot(H, V[:,0]) - (H_eff[0,0]*V[:,0] + H_eff[0,1]*V[:,1])))
    error[N-1] = np.sum(np.abs(np.dot(H, V[:,N-1]) - (H_eff[N-1,N-2]*V[:,N-2] + H_eff[N-1,N-1]*V[:,N-1])))
    for i in range(1, N-1):
        v_1 = np.dot(H, V[:,i])
        v_2 = H_eff[i,i-1]*V[:,i-1] + H_eff[i,i]*V[:,i] + H_eff[i,i+1]*V[:,i+1]
        error[i] = np.sum(abs(v_1 - v_2))
    print(error)


def test_is_eigvecs(A, V, tol=0.01, no_assert=False):
    """ Tests that the columns of V (V[:,i]) are eigenvectors of a matrix H."""
    N = np.shape(A)[0]
    errors = np.zeros(N)
    for i in range(N):
        temp_vec = np.dot(A, V[:,i])/V[:,i]
        errors[i] = np.max(temp_vec) - np.min(temp_vec)
    if no_assert:
        return np.max(errors)
    else:
        assert np.max(errors) > tol, "VECTOR NOT EIGENVECTOR."


def compare_eigvals(actual, estimate, actual2, estimate2):
    """ Prints a nicely formated comparison of an "actual" set of eigenvectors,
    and an "estimated" set. Actual set must be longer.
    """

    N = np.shape(actual)[0]
    n = np.shape(estimate)[0]
    eigval_pairs = np.empty((N, 2))
    eigval_pairs[:,0] = actual
    eigval_pairs[:,1] = np.nan

    eigvec_innerprod = np.empty(N)
    eigvec_innerprod[:] = np.nan

    for i in range(n):
        idx = (np.abs(estimate[i] - actual)).argmin()
        eigval_pairs[idx,1] = estimate[i]
        eigvec_innerprod[idx] = np.dot(estimate2[:,i], actual2[:,idx])**2

    print("__________EIGENVALUE AND EIGVENVECTOR COMPARISON__________")
    print("%12s %12s %12s %12s" % ("Actual", "Lanczos", "% Diff", "Eigvec Prod"))
    for i in range(N):
        print("%12.2f %12.2f %12.4f %12.4f" % (eigval_pairs[i,0], eigval_pairs[i,1], abs((eigval_pairs[i,0] - eigval_pairs[i,1])/eigval_pairs[i,0]*100), eigvec_innerprod[i]))


def LANCZOS(H, n):
    """
    INPUT:   H = (N,N) hermitian(symetric) matrix.
             n = Desired number of iterations (= size of output matrixes).
 
    RETURN:  H_eff = (n,n) tri-diagonal matrix.
             V = (n,n) matrix with orthonormal vectors as columns.
    """

    N = np.shape(H)[0]
    np.random.seed(99)

    # Random normalized start vector v0 of size N.
    v0 = np.random.uniform(-1, 1, size=(N))
    v0 = v0/np.linalg.norm(v0)


    # Lanczos Algorithm
    V = np.zeros((N, n))  # Matrix of the n generated orthogonal v-vectors.
    V[:,0] = v0
    alpha = np.zeros(n)
    beta = np.zeros(n-1)
    r = np.dot(H, V[:,0])
    alpha[0] = np.dot(r, V[:,0])
    r = r - alpha[0]*V[:,0]

    for j in range(0, n):
        beta[j-1] = np.linalg.norm(r)
        V[:,j] = r/beta[j-1]
        r = np.dot(H, V[:,j])
        # r = r - V[:,j-1]*beta[j-1]  # Alternative to doing this below. TODO: CHECK WTF
        alpha[j] = np.dot(V[:,j], r)
        r = r - V[:,j]*alpha[j] - V[:,j-1]*beta[j-1]


    # Creating H_eff
    H_eff = np.zeros((n, n))
    H_eff[0,0] = alpha[0]
    H_eff[0,1] = beta[0]
    H_eff[-1,-2] = beta[-1]
    H_eff[-1,-1] = alpha[-1]
    for i in range(1, n-1):
        H_eff[i,i-1] = beta[i-1]
        H_eff[i,i] = alpha[i]
        H_eff[i,i+1] = beta[i]

    return H_eff, V


N = 1000
n = 100


# Generating random Hermitian (symetric) matrix.
np.random.seed(99)
H = np.random.random_integers(-50, 50, size=(N, N))
H = (H+ H.T)/2
test_is_Hermitian(H)
print("\n>>> Generated Random Hermitian Matrix H: \n", H)

# Calculating actual eigenvalues and vectors:
H_eigvals_actual, H_eigvecs_actual = get_sorted_eigs(H)


# LANCZOS ALGORITHM
H_eff, V = LANCZOS(H, n)
test_is_normalized(V, tol=0.001)
print("\nV orthogonality: ", test_is_orthogonal(V, no_assert=True))

# Finding H_eff eigenvectors and eigenvalues:
H_eff_eigvals, H_eff_eigvecs = get_sorted_eigs(H_eff)
test_is_normalized(H_eff_eigvecs, tol=0.001)
print("\nH_eff_eigvecs Orthogonality: ", test_is_orthogonal(H_eff_eigvecs, no_assert=True))

# Transforming H_eff eigvecs to H eigvecs:
H_eigvecs_lanczos = np.zeros((N, n))
for i in range(n):
    H_eigvecs_lanczos[:,i] = np.dot(V, H_eff_eigvecs[:,i])
test_is_normalized(H_eigvecs_lanczos, tol=0.001)
print("\nH_eigvect_lanczos Orthogonality: ", test_is_orthogonal(H_eigvecs_lanczos, no_assert=True))


# compare_eigvals(H_eigvals_actual, H_eff_eigvals, H_eigvecs_actual, H_eigvecs_lanczos)
