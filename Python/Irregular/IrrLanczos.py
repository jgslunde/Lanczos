import numpy as np
import math as m
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.sparse
from tqdm import tqdm
import time
import sys


class IrrLanczos:
    """ Class implementation of the Lanczos algorithm.
        Takes a Hermitian matrix H in the constructor.
        To use, run
            execute_Lanczos()
        and
            get_H_eigs().
    """
    def __init__(self, H):
        # self.test_is_Hermitian(H)
        self.H = H
        self.M = np.shape(H)[0]
        self.Lanczos_has_been_executed = False

        self.H_eigs_have_been_found = False
        self.H_exact_eigs_have_been_found = False

    @property
    def H_eff(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")
        else:
            return self._H_eff

    @property
    def V1(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")
        else:
            return self._V1

    @property
    def V2(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")
        else:
            return self._V2

    @property
    def H_eigvecs(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        return self._H_eigvecs

    @property
    def H_eigvecs1(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        return self._H_eigvecs1

    @property
    def H_eigvecs2(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        return self._H_eigvecs2

    @property
    def H_eigvals(self):
        if not self.H_eigs_have_been_found:
            self.get_H_eigs()
        return self._H_eigvals

    @property
    def H_eigvals_actual(self):
        if not self.H_exact_eigs_have_been_found:
            self.find_exact_eigs()
            self.H_exact_eigs_have_been_found = True
        return self._H_eigvals_actual

    @property
    def H_eigvecs_actual(self):
        if not self.H_exact_eigs_have_been_found:
            self.find_exact_eigs()
            self.H_exact_eigs_have_been_found = True
        return self._H_eigvecs_actual

    def find_exact_eigs(self, nr_vecs=20):
        print("+++ Calculating exact eigs using scipy.sparse.linalg.eigsh.")
        self._H_eigvals_actual, self._H_eigvecs_actual = scipy.sparse.linalg.eigsh(self.H, k=nr_vecs, which="SM")
        print("+++ Finished calculating exact eigs.")



    def execute_Lanczos(self, n, seed=99, use_cuda=True, v0=None):
        if n > self.M:
            raise ValueError("n cannot be larger than M!")

        print("+++ Executing Lanczos algorithm")
        self.n = n

        H = self.H
        M = self.M

        if use_cuda:
            import numpy as np
            H = cupyx.scipy.sparse.csc_matrix(H, dtype=np.float64)
            import cupy as np
        else:
            import numpy as np

        np.random.seed(seed)
        HT = np.transpose(H)

        # Random normalized start vector v0 of size N.
        if v0 is None:
            v0 = np.random.uniform(-1, 1, size=(M))
        else:
            v0 = np.array(v0)
        v0 = v0/np.linalg.norm(v0)

        # Lanczos Algorithm
        # NOTE: the v-vectors in V are ROW VECTORS within this function, for cache reasons, and is transposed to COLUMN VECTORS upon finishing.
        V1 = np.zeros((n, M))  # Matrix of the n generated orthogonal v-vectors. 
        V1[0] = v0
        V2 = np.zeros((n, M))
        V2[0] = v0
        alpha = np.zeros(n)
        beta = np.zeros(n-1)
        gamma = np.zeros(n-1)
        w = np.zeros(n)
        r = H*V1[0]
        s = HT*V2[0]

        for j in tqdm(range(0, n-1)):
            alpha[j] = np.dot(V1[j], r)
            r = r - alpha[j]*V1[j]
            s = s - alpha[j]*V2[j]
            w[j] = np.dot(r, s)
            # print(w[j])
            beta[j] = m.sqrt(w[j])
            gamma[j] = w[j]/beta[j]
            V1[j+1] = r/beta[j]
            V2[j+1] = s/gamma[j]
            # REORTHOGONALIZATION:
            # self.bireorthogonalize(V1, V2, j, use_cuda=use_cuda)
            # print("ASDFASDF ", np.linalg.norm(V1[j]), np.linalg.norm(V2[j]))
            asdf = 0
            # for i in range(j):
            #     asdf += np.dot(V1[:5], V1[j].T)
            # print(asdf)

            r = H*V1[j+1]
            s = HT*V2[j+1]
            r = r - V1[j]*gamma[j]
            s = s - V2[j]*beta[j]
        alpha[n-1] = np.dot(V1[n-1], r)

        # Creating H_eff
        H_eff = np.zeros((n, n))
        H_eff[0,0] = alpha[0]
        H_eff[0,1] = gamma[0]
        H_eff[-1,-2] = beta[-1]
        H_eff[-1,-1] = alpha[-1]
        for i in tqdm(range(1, n-1)):
            H_eff[i,i-1] = beta[i-1]
            H_eff[i,i] = alpha[i]
            H_eff[i,i+1] = gamma[i-1]

        if use_cuda:
            # If having used cuda, convert H_eff and V to numpy objects, and H to scipy.sparse.
            import numpy as np
            self._H_eff = cp.asnumpy(H_eff)
            self._V1 = cp.asnumpy(V1.T)
            self._V2 = cp.asnumpy(V2.T)
            self.H = H.get()
        else:
            self._H_eff, self._V1, self.V2 = H_eff, V1.T, V2.T
        print("+++ Lanczos executed successfully.")
        self.Lanczos_has_been_executed = True





    def execute_LanczosOld(self, n, seed=99, use_cuda=True, v0=None):
        if n > self.M:
            raise ValueError("n cannot be larger than M!")

        print("+++ Executing Lanczos algorithm")
        self.n = n

        H = self.H
        M = self.M

        if use_cuda:
            import numpy as np
            H = cupyx.scipy.sparse.csc_matrix(H,dtype=np.float64)
            import cupy as np
        else:
            import numpy as np

        np.random.seed(seed)

        # Random normalized start vector v0 of size N.
        if v0 is None:
            v0 = np.random.uniform(-1, 1, size=(M))
        else:
            v0 = np.array(v0)
        v0 = v0/np.linalg.norm(v0)

        # Lanczos Algorithm
        # NOTE: the v-vectors in V are ROW VECTORS within this function, for cache reasons, and is transposed to COLUMN VECTORS upon finishing.
        V = np.zeros((n, M))  # Matrix of the n generated orthogonal v-vectors. 
        V[0] = v0
        alpha = np.zeros(n)
        beta = np.zeros(n-1)
        r = H*V[0]
        # r = HT*r
        alpha[0] = np.dot(r, V[0])
        r = r - alpha[0]*V[0] 
        for j in tqdm(range(0, n)):
            beta[j-1] = np.linalg.norm(r)
            V[j] = r/beta[j-1]
            # REORTHOGONALIZATION:
            self.reorthogonalize(V, j, use_cuda=use_cuda)
            r = H*V[j]
            # r = HT*r
            # r = r - V[:,j-1]*beta[j-1]  # Alternative to doing this below. TODO: check.
            alpha[j] = np.dot(V[j], r)
            r = r - V[j]*alpha[j] - V[j-1]*beta[j-1]

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

        if use_cuda:
            # If having used cuda, convert H_eff and V to numpy objects, and H to scipy.sparse.
            import numpy as np
            self._H_eff = cp.asnumpy(H_eff)
            self._V = cp.asnumpy(V.T)
            self.H = H.get()
        else:
            self._H_eff, self._V = H_eff, V.T
        print("+++ Lanczos executed successfully.")
        self.Lanczos_has_been_executed = True



    def get_H_eigsOld(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")

        print("+++ Converting eigenvectors from H_eff to H basis.")
        M, n, V = self.M, self.n, self._V
        H_eff_eigvals, H_eff_eigvecs = np.linalg.eigh(self.H_eff)
        
        # Transforming H_eff eigvecs to H eigvecs:
        H_eigvecs_lanczos = np.zeros((M, n))
        for i in range(n):
            H_eigvecs_lanczos[:,i] = np.dot(V, H_eff_eigvecs[:,i])
        self.test_is_normalized(H_eigvecs_lanczos, tol=0.001)
        # self.test_is_orthogonal(H_eigvecs_lanczos, tol=0.01)
        
        self._H_eigvals = H_eff_eigvals
        self._H_eigvecs = H_eigvecs_lanczos
        print("+++ Finished Converting.")
        self.H_eigs_have_been_found = True


    def get_H_eigs(self):
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")

        print("+++ Converting eigenvectors from H_eff to H basis.")
        M, n, V1, V2 = self.M, self.n, self.V1, self.V2
        H_eff_eigvals, H_eff_eigvecs = np.linalg.eigh(self.H_eff)
        
        # Transforming H_eff eigvecs to H eigvecs:
        H_eigvecs_lanczos1 = np.zeros((M, n))
        H_eigvecs_lanczos2 = np.zeros((M, n))
        for i in range(n):
            H_eigvecs_lanczos1[:,i] = np.dot(V1, H_eff_eigvecs[:,i])
            H_eigvecs_lanczos2[:,i] = np.dot(V2, H_eff_eigvecs[:,i])
        self.test_is_normalized(H_eigvecs_lanczos1, tol=0.001)
        self.test_is_normalized(H_eigvecs_lanczos2, tol=0.001)
        self.test_is_orthogonal(H_eigvecs_lanczos1, tol=0.01)
        self.test_is_orthogonal(H_eigvecs_lanczos2, tol=0.01)
        
        
        self._H_eigvals = H_eff_eigvals
        self._H_eigvecs1 = H_eigvecs_lanczos1
        self._H_eigvecs2 = H_eigvecs_lanczos2
        print("+++ Finished Converting.")
        self.H_eigs_have_been_found = True


    def print_good_eigsOld(self, tol=0.01, print_nr=20, print_bad=True, normal_eq=False):
        """ Prints out found H-eigs that actually match well, requiring Hx = hx => (Hx)/|Hx| = x within a tolerance."""
        H, eigvecs, eigvals = self.H, self.H_eigvecs, self.H_eigvals
        inner_prod = np.zeros(self.n)
        for i in range(self.n):
            eigvec = eigvecs[:,i]
            eigval = eigvals[i]
            Hv = H*eigvec
            Hv = Hv/np.linalg.norm(Hv)
            inner_prod[i] = np.dot(Hv, eigvec)**2

        if normal_eq:
            eigvals = np.sqrt(eigvals)

        print("__________EIGENVALUE AND EIGVENVECTOR COMPARISON__________")
        print("%12s %12s" % ("Eigval", "Eigvec InnerProd"))
        for i in range(print_nr):
            if abs(1 - inner_prod[i]) < tol:
                print(f"\033[92m\033[1m{eigvals[i]:12.4f}{inner_prod[i]:12.6f} \033[0m")
            else:
                print(f"\033[33m{eigvals[i]:12.4f}{inner_prod[i]:12.6f} --- BAD\033[0m")
    

    def print_good_eigs(self, tol=0.01, print_nr=20, print_bad=True, normal_eq=False):
        """ Prints out found H-eigs that actually match well, requiring Hx = hx => (Hx)/|Hx| = x within a tolerance."""
        H, eigvecs1, eigvecs2, eigvals = self.H, self.H_eigvecs1, self.H_eigvecs2, self.H_eigvals
        inner_prod = np.zeros(self.n)
        for i in range(self.n):
            eigvec = eigvecs[:,i]
            eigval = eigvals[i]
            Hv = H*eigvec
            Hv = Hv/np.linalg.norm(Hv)
            inner_prod[i] = np.dot(Hv, eigvec)**2

        if normal_eq:
            eigvals = np.sqrt(eigvals)

        print("__________EIGENVALUE AND EIGVENVECTOR COMPARISON__________")
        print("%12s %12s" % ("Eigval", "Eigvec InnerProd"))
        for i in range(print_nr):
            if abs(1 - inner_prod[i]) < tol:
                print(f"\033[92m\033[1m{eigvals[i]:12.4f}{inner_prod[i]:12.6f} \033[0m")
            else:
                print(f"\033[33m{eigvals[i]:12.4f}{inner_prod[i]:12.6f} --- BAD\033[0m")



    def compare_eigs(self):
        """ Prints a nicely formated comparison of the actual and Lanczos-estimated eigenvalues and eigenvectors, matched after max inner product with actual eigenvectors. Note that this will only really work if H is small, and exact eigenvectors can be solved with numpy.
        """
        if not self.Lanczos_has_been_executed:
            raise ValueError("Lanczos Algorithm has not been called.")

        print("+++ Comparing to exact eigs.")
        eigval_actual, eigvec_actual = self.H_eigvals_actual, self.H_eigvecs_actual
        eigval_estimate, eigvec_estimate = self.H_eigvals, self.H_eigvecs

        N, n, nr_vecs = self.M, self.n, len(eigval_actual)
        eigval_pairs = np.empty((nr_vecs, 2))
        eigval_pairs[:,0] = eigval_actual
        eigval_pairs[:,1] = np.nan

        eigvec_innerprod = np.empty(nr_vecs)
        eigvec_innerprod[:] = np.nan

        idx_pairs = np.zeros(nr_vecs)
        idx_pairs[:] = np.nan
        for i in range(n):
            idx = (np.dot(eigvec_estimate[:,i], eigvec_actual)**2).argmax()
            inner_prod = np.dot(eigvec_estimate[:,i], eigvec_actual[:,idx])**2
            if np.isnan(eigvec_innerprod[idx]) or (inner_prod > eigvec_innerprod[idx]): # If there isn't already a better match, or if there is no match yet.
                eigval_pairs[idx,1] = eigval_estimate[i]
                eigvec_innerprod[idx] = inner_prod
                idx_pairs[idx] = i
                    
        perc_diff_eigval = abs((eigval_pairs[:,0] - eigval_pairs[:,1])/eigval_pairs[:,1])*100



    @staticmethod
    def bireorthogonalize(V1, V2, j, use_cuda=True):
        if use_cuda:
            if j != 0:
                inner_prods_uv = cp.sum(V1[j,:]*V1[:j,:], axis=1)
                inner_prods_uu = cp.sum(V1[:j,:]*V1[:j,:],axis=1)
                V1[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V1[:j,:],axis=0)

                inner_prods_uv = cp.sum(V1[j,:]*V2[:j,:], axis=1)
                inner_prods_uu = cp.sum(V2[:j,:]*V2[:j,:],axis=1)
                V1[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V2[:j,:],axis=0)

                inner_prods_uv = cp.sum(V2[j,:]*V2[:j,:], axis=1)
                inner_prods_uu = cp.sum(V2[:j,:]*V2[:j,:],axis=1)
                V2[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V2[:j,:],axis=0)

                inner_prods_uv = cp.sum(V2[j,:]*V1[:j,:], axis=1)
                inner_prods_uu = cp.sum(V1[:j,:]*V1[:j,:],axis=1)
                V2[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V1[:j,:],axis=0)
        else:
            raise NotImplementedError("sorry :)")

    @staticmethod
    def reorthogonalize(V, j, use_cuda=True):
        """ Reorthogonalizes element number i in V matrix."""
        # Notes. The commented out method should in theory be more optimal, as it only computes the part of the products actually used.
        # However, it has memory stability issues.
        if use_cuda:
            inner_prods = cp.sum(V[j]*V, axis=1)
            V[j] = 2*V[j] - cp.sum(inner_prods[:,None]*V, axis=0)
            # if j != 0:
            #     inner_prods_uv = cp.sum(V[j,:]*V[:j,:], axis=1)
            #     inner_prods_uu = cp.sum(V[:j,:]*V[:j,:],axis=1)
            #     V[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V[:j,:],axis=0)
            # for i in range(j): # Old loop implementation
            #     V[:,j] = V[:,j] - cp.dot(V[:,i], V[:,j])*V[:,i]
        else:
            inner_prods = np.sum(V[j]*V, axis=1)
            V[j] = 2*V[j] - np.sum(inner_prods[:,None]*V, axis=0)
            # for i in range(j):
                # V[:,j] = V[:,j] - np.dot(V[:,i], V[:,j])*V[:,i]

    @staticmethod
    def get_matched_eigs(v, vL, l, lL):
        """From a set of N analytical eigenvectors and values v and l, and a set of n estimated eigenvectors and values vL and lL,
            return a ordered set of eigenvectors and values (v_ordered, vL_ordered, l_ordered, lL_ordered), all of length n, where
            the ordering is determined by the correlation between the set of eigenvectors, such that the best fitting eigenvectors are at idex 0."""
        n = len(lL)
        N = len(l)

        eigvec_innerprod = np.zeros(n)
        map_v2vL = np.zeros(N, dtype=int)
        map_vL2v = np.zeros(n, dtype=int)
        map_vL2v[:] = np.nan
        map_v2vL[:] = np.nan

        for i in range(n):
            idx = (np.dot(vL[:,i], v)**2 ).argmax()
            map_v2vL[idx] = i
            map_vL2v[i] = idx
            eigvec_innerprod[i] = np.dot(vL[:,i], v[:,idx])**2
        sort_idx = eigvec_innerprod.argsort()[::-1]

        vL_sorted = vL[:,sort_idx]
        lL_sorted = lL[sort_idx]
        v_sorted = v[:, map_vL2v[sort_idx] ]
        l_sorted = l[map_vL2v[sort_idx]]
        
        return v_sorted, vL_sorted, l_sorted, lL_sorted


    @staticmethod
    def test_is_Hermitian(A):
        """Tests if A is Hermitian."""
        assert (A == A.A).all(), "A IS NOT HERMITIAN!"


    @staticmethod
    def test_is_normalized(V, tol=0.001, no_assert=False):
        """ INPUT: V = (M,m) matrix. Tests that it's columns V[:,i] are normal.
            no_assert = Setting this to True will NOT run a test, and instead
                                return the inner product of the least normal vector.
        """
        N = np.shape(V)[1]
        normalizations = np.zeros(N)
        for i in range(N):
            normalizations[i] = np.linalg.norm(V[:,i])
        idx = np.argmin(np.abs(normalizations - 1))

        if no_assert:
            return normalizations[idx]
        else:
            assert np.abs(normalizations[idx]-1) < tol,\
            "VECTOR HAS NORM %.4f. IS NOT NORMALIZED." % normalizations[idx]


    @staticmethod
    def test_is_orthogonal(V, tol=0.01, no_assert=False):
        """ INPUT: V = (M,m) matrix of normal column vectors.
            Tests whether the column vectors of V[:,i] are all orthogonal.
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