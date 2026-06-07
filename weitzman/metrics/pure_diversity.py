"""
Pure Diversity (PD) indicator for Euclidean distance.

This file is a Python adaptation of the MATLAB implementation distributed with:

    Handing Wang, Yaochu Jin, Xin Yao,
    "Diversity Assessment in Many-Objective Optimization",
    IEEE Transactions on Cybernetics, 2017.
    DOI: 10.1109/TCYB.2016.2550502

Original MATLAB code:
    https://github.com/HandingWang/PD

Authors state that the code is free to use for non-commercial purposes
and requests acknowledgement. This Python adaptation preserves that
restriction.
"""

import numpy as np
from numpy.typing import NDArray


def _connected(C: NDArray[np.int_], i: int, j: int) -> bool:
    """Return True if nodes i and j are connected in the graph C."""
    children = np.where(C[i, :] == 1)[0]
    if np.any(children == j):
        return True

    C_copy = C.copy()
    C_copy[i, :] = 0
    C_copy[:, i] = 0

    for c in children:
        if _connected(C_copy, c, j):
            return True

    return False


def PD(X: NDArray[np.float64]) -> float:
    """
    Compute the Pure Diversity (PD) value for a population X of shape (n, d).

    PD builds a spanning structure by iteratively selecting the pair
    (i, nearest-neighbour of i) with the largest nearest-neighbour distance,
    accumulates that distance, and removes i from further consideration.
    Pairs already connected in the growing graph are skipped.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    C = np.zeros((n, n), dtype=int)    # connection matrix (grows as pairs are linked)
    D = np.zeros((n, n), dtype=float)  # pairwise L2 dissimilarity matrix

    for i in range(n - 1):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[j] - X[i])
            D[i, j] = d
            D[j, i] = d

    DMAX = np.max(D) + 1.0
    np.fill_diagonal(D, DMAX)   # a point is never its own nearest neighbour

    pd = 0.0

    for _ in range(n - 1):
        d = np.min(D, axis=1)
        J = np.argmin(D, axis=1)

        i = int(np.argmax(d))
        dmx = d[i]

        # If i and J[i] are already connected, invalidate and find next candidate.
        while _connected(C, i, J[i]):
            D[J[i], i] = DMAX
            D[i, J[i]] = DMAX
            d = np.min(D, axis=1)
            J = np.argmin(D, axis=1)
            i = int(np.argmax(d))
            dmx = d[i]

        C[J[i], i] = 1
        C[i, J[i]] = 1
        pd += dmx

        D[J[i], i] = DMAX
        D[i, :] = -1   # mark i as consumed - it will never be selected again

    return pd
