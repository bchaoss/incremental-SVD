# Implementation of incremental SVD algorithm (iSVD for short) from [Brand, 2002]
# Time: 2023/12/26
# Author: bchaoss
# Reference: Brand, Matthew. “Incremental Singular Value Decomposition of Uncertain Data with Missing Values.” European Conference on Computer Vision (2002).


import numpy as np
from numpy import matmul, sqrt, block, shape, eye, zeros, hstack, vstack
from numpy.linalg import svd, norm
from numpy.random import randn


def initializeISVD(u1, W):
    S = sqrt(u1.T @ W @ u1)
    Q = u1 / S
    R = eye(1, 1)

    S *= eye(1, 1)
    Q = Q.reshape((Q.shape[0], 1))
    return Q, S, R


# reOrthogonalization
def modified_gram_schmidt(Q, W, tol):
    if np.abs(Q[:, -1].T @ W @ Q[:, 0]) > tol:
        k = Q.shape[1]
        for i in range(k):
            a = Q[:, i]
            for j in range(i):
                Q[:, i] = Q[:, i] - ((a.T @ W @ Q[:, j]) /
                                     (Q[:, j].T @ W @ Q[:, j])) * Q[:, j]
            norm = np.sqrt(Q[:, i].T @ W @ Q[:, i])
            Q[:, i] = Q[:, i] / norm
    return Q


def updateISVD(Q, S, R, u_l, W, tol):
    d = Q.T @ W @ u_l
    if not shape(d):
        d *= eye(1, 1)
    e = u_l - Q @ d
    p = sqrt(e.T @ W @ e) * eye(1, 1)

    if p < tol:
        p = zeros((1, 1))
    else:
        e = e / p[0, 0].item()

    k = shape(S)[0] if shape(S) else 1
    Y = vstack((hstack((S, d)), hstack((zeros((1, k)), p))))
    Qy, Sy, Ry = svd(Y, full_matrices=True, compute_uv=True)
    Sy = np.diag(Sy)

    l = shape(R)[0]
    if p < tol:
        Q = Q @ Qy[:k, :k]
        S = Sy[:k, :k]
        R = vstack((hstack((R, zeros((l, 1)))), hstack(
            (zeros((1, k)), eye(1))))) @ Ry[:, :k]
    else:
        Q = hstack((Q, e)) @ Qy
        S = Sy
        R = vstack((hstack((R, zeros((l, 1)))),
                   hstack((zeros((1, k)), eye(1))))) @ Ry

    return Q, S, R


# main algo
def iSVD(U, W, tol=1e-15):
    m, n = shape(U)[0], shape(U)[1]
    u_0 = U[:, 0].reshape((m, 1))
    Q, S, R = initializeISVD(u_0, W)
    for i in range(1, n):
        u_l = U[:, i].reshape((m, 1))
        Q, S, R = updateISVD(Q, S, R, u_l, W, tol)
        Q = modified_gram_schmidt(Q, W, tol)
    return Q, S, R


U = randn(30, 10) @ randn(10, 20)

m = shape(U)[0]
W = eye(m, m)

Q, S, R = iSVD(U=U, W=W)
