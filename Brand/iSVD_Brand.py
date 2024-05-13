# Implementation of incremental SVD algorithm (iSVD for short) from [Brand, 2002]
# Time: 2024/5/13
# Author: bchaoss
# Reference: Brand, Matthew. “Incremental Singular Value Decomposition of Uncertain Data with Missing Values.” European Conference on Computer Vision (2002).


import numpy as np
from numpy import matmul, sqrt, block, shape, eye, zeros, hstack, vstack
from numpy.linalg import svd, norm
from numpy.random import randn

# reOrthogonalization


def modified_gram_schmidt(U, tol):
    if np.abs(U[:, -1].T @ U[:, 0]) > tol:
        k = U.shape[1]
        for i in range(k):
            a = U[:, i]
            for j in range(i):
                U[:, i] = U[:, i] - ((a.T @ U[:, j]) /
                                     (U[:, j].T @ U[:, j])) * U[:, j]
            norm = np.sqrt(U[:, i].T @ U[:, i])
            U[:, i] = U[:, i] / norm
    return U


def initializeISVD(u1):
    S = sqrt(u1.T @ u1)
    U = u1 / S
    V = eye(1, 1)

    S *= eye(1, 1)
    U = U.reshape((U.shape[0], 1))
    return U, S, V


def updateISVD(U, S, V, a_l, tol):
    d = U.T @ a_l
    if not shape(d):
        d *= eye(1, 1)
    e = a_l - U @ d
    p = sqrt(e.T @ e) * eye(1, 1)

    if p < tol:
        p = zeros((1, 1))
    else:
        e = e / p[0, 0].item()

    k = shape(S)[0] if shape(S) else 1
    Y = vstack((hstack((S, d)), hstack((zeros((1, k)), p))))
    Uy, Sy, Vy = svd(Y, full_matrices=True, compute_uv=True)
    Sy = np.diag(Sy)

    l = shape(V)[0]
    if p < tol:
        U = U @ Uy[:k, :k]
        S = Sy[:k, :k]
        V = vstack((hstack((V, zeros((l, 1)))), hstack(
            (zeros((1, k)), eye(1))))) @ Vy[:, :k]
    else:
        U = hstack((U, e)) @ Uy
        S = Sy
        V = vstack((hstack((V, zeros((l, 1)))),
                   hstack((zeros((1, k)), eye(1))))) @ Vy

    return U, S, V


# main algo
def iSVD(A, U=None, S=None, V=None, compute_V=True):
    tol = 1e-09
    m, n = shape(A)[0], shape(A)[1]
    flag_append = False

    if U is not None and S is not None and V is not None:
        if shape(U)[0] != m:
            raise "got a wrong dimension."
        start_A = 0
        compute_V = False
    else:
        U, S, V = initializeISVD(A[:, 0].reshape((m, 1)))
        start_A = 1

    for i in range(start_A, n):
        a_l = A[:, i].reshape((m, 1))
        U, S, V = updateISVD(U, S, V, a_l, tol)
        U = modified_gram_schmidt(U, tol)

    if compute_V:
        Vt = np.linalg.lstsq(S, U.T @ A, rcond=None)[0]
        return U, S, Vt
    return U, S
