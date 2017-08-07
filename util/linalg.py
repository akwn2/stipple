"""
linalg.py
linear algebra utilities including matrix factorisation, slicing and blocking.
"""
import numpy as np
import scipy as sp
import scipy.linalg as linalg


def qform(x, A, y):
    """
    convenience function to calculate the quadratic form x.T A y
    :param x: first vector
    :param A: middle matrix
    :param y: last vector
    :return: scalar
    """
    return np.dot(x.T, np.dot(A, y))


# Cholesky factorisation utilities
def rchol(x, attempts=6):
    """
    robust cholesky
    :param x: the matrix to be factorised
    :param attempts: maximum number of attempts
    :return:
    """
    jitter = 0
    try:
        L = linalg.cholesky(x, lower=True, check_finite=False)
        return L, jitter
    except sp.linalg.LinAlgError:
        a = 0
        jitter = 1e-6 * sp.mean(sp.diag(x))
        while a < attempts:
            try:
                L = linalg.cholesky(x + jitter * sp.eye(x.shape[0]), lower=True, check_finite=False)
                return L, jitter
            except sp.linalg.LinAlgError:
                jitter *= 10.

    raise sp.linalg.LinAlgError


def solve_chol(L, A):
    """
    solve Lx = A
    :param L: Cholesky factor
    :param X: RHS matrix
    :return:
    """
    LiA = linalg.solve_triangular(L, A, lower=True, check_finite=False)
    return LiA


def inv_mult_chol(L, X):
    """
    calculate dot(K_inv, X) for K_inv the inverse of K = L * L'
    :param L: cholesky factor of K
    :param X: matrix or vector to be multiplied by the inverse of K
    :return:
    """
    LiX = linalg.solve_triangular(L, X, lower=True, check_finite=False)
    KiX = linalg.solve_triangular(L.T, LiX, lower=False, check_finite=False)
    return KiX


def invc(L):
    """
    find the inverse of K = L * L' using cholesky factor L
    :param L: cholesky factor of K
    :return:
    """
    return inv_mult_chol(L, sp.eye(L.shape[0]))


def logdetK(L):
    """
    numerical stable way to calculates the determinant of K using its cholesky factor L
    :param L: cholesky factor of K
    :return: log of determinant of K
    """
    return 2. * sum(np.log(np.diag(L)))


# Block matrices
def blockm(A, B, C, D):
    """
    creates a block matrix [[A, B],[C, D]] as a 2D array
    :param A: upper left block
    :param B: upper right block
    :param C: lower left block
    :param D: lower right block
    :return:
    """

    return np.asarray(np.bmat([[A, B], [C, D]]))


def get_sym_blocks(K):
    """
    gets the N x N blocks of a 2N x 2N symmetric matrix K
    :param K: 2N x 2N matrix
    :return:
    """
    N = K.shape[0] / 2

    return K[0:N, 0:N], K[0:N, N:], K[N:, N:]
