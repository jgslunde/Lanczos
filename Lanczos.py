import numpy as np
import cupy as cp
from tqdm import tqdm

class Lanczos:
    def __init__(self, H):
        self.test_is_Hermitian(H)
        self.H = H
        self.N = np.shape(H)[0]
        self.Lanczos_has_been_executed = False
        self.H_eigs_have_been_found = False

    @property
    def H_eff(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")
        else:
            return self._H_eff

    @property
    def V(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")
        else:
            return self._V

    @property
    def H_eigvecs(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        return self._H_eigvecs

    @property
    def H_eigvals(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        else:
            return self._H_eigvals


    def execute_Lanczos(self, n, seed=99):
        print("+++Executing Lanczos algorithm")
        self.n = n
        H = self.H
        N = self.N
        np.random.seed(seed)

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
        for j in tqdm(range(0, n)):
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
        for i in tqdm(range(1, n-1)):
            H_eff[i,i-1] = beta[i-1]
            H_eff[i,i] = alpha[i]
            H_eff[i,i+1] = beta[i]

        self._H_eff, self._V = H_eff, V
        print("+++Lanczos executed successfully.")
        self.Lanczos_has_been_executed = True


    def get_H_eigs(self):
        print("+++ Converting eigenvectors from H_eff to H basis.")
        N, n, V = self.N, self.n, self.V
        H_eff_eigvals, H_eff_eigvecs = np.linalg.eigh(self.H_eff)
        
        # Transforming H_eff eigvecs to H eigvecs:
        H_eigvecs_lanczos = np.zeros((N, n))
        for i in range(n):
            H_eigvecs_lanczos[:,i] = np.dot(V, H_eff_eigvecs[:,i])
        self.test_is_normalized(H_eigvecs_lanczos, tol=0.001)
        self.test_is_orthogonal(H_eigvecs_lanczos, tol=0.01)
        
        self._H_eigvals = H_eff_eigvals
        self._H_eigvecs = H_eigvecs_lanczos
        print("+++ Finished Converting.")
        self.H_eigs_have_been_found = True

    
    def compare_eigs(self):
        """ Prints a nicely formated comparison of the actual and Lanczos-estimated eigenvalues and eigenvectors.
        """
        eigval_actual, eigvec_actual = np.linalg.eigh(self.H)
        eigval_estimate, eigvec_estimate = self.H_eigvals, self.H_eigvecs

        N, n = self.N, self.n
        eigval_pairs = np.empty((N, 2))
        eigval_pairs[:,0] = eigval_actual
        eigval_pairs[:,1] = np.nan

        eigvec_innerprod = np.empty(N)
        eigvec_innerprod[:] = np.nan

        for i in range(n):
            idx = (np.abs(eigval_estimate[i] - eigval_actual)).argmin()
            eigval_pairs[idx,1] = eigval_estimate[i]
            eigvec_innerprod[idx] = np.dot(eigvec_estimate[:,i], eigvec_actual[:,idx])**2

        print("__________EIGENVALUE AND EIGVENVECTOR COMPARISON__________")
        print("%12s %12s %12s %12s" % ("Actual", "Lanczos", "% Diff", "Eigvec Prod"))
        for i in range(N):
            print("%12.2f %12.2f %12.4f %12.4f" % (eigval_pairs[i,0], eigval_pairs[i,1], abs((eigval_pairs[i,0] - eigval_pairs[i,1])/eigval_pairs[i,0]*100), eigvec_innerprod[i]))


    @staticmethod
    def test_is_Hermitian(A):
        """Tests if A is Hermitian."""
        assert (np.matrix(A) == np.matrix(A).A).all(),\
        "A IS NOT HERMITIAN!"


    @staticmethod
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


    @staticmethod
    def test_is_orthogonal(V, tol=0.01, no_assert=False):
        """ INPUT: V = (M,m) matrix. Tests whether the column vectors of V[:,i] are all orthogonal.
            no_assert = Setting this to True will NOT run a test, and instead
                                return the largest inner product between two vectors.
        """
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


    @staticmethod
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


if __name__ == "__main__":
    import numpy as np

    N = 20
    n = 20

    # Generating random Hermitian (symetric) matrix.
    np.random.seed(99)
    H = np.random.random_integers(-50, 50, size=(N, N))
    H = (H+ H.T)/2

    TEST = Lanczos(H)
    TEST.execute_Lanczos(n)
    TEST.get_H_eigs()

    H_eigvals_actual, H_eigvecs_actual = np.linalg.eigh(H)
    TEST.compare_eigs()