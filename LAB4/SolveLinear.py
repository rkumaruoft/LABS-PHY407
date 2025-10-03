# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman

# The following will be useful for partial pivoting
from numpy import array, empty, copy as np
import numpy as np


def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


def PartialPivot(A_in, v_in, m):
    """Perform partial pivoting on column m without modifying inputs"""

    A_copy = np.copy(A_in).astype(float)
    v_copy = np.copy(v_in).astype(float)
    N = A_copy.shape[0]

    #Find pivot row: index of max abs value in column m at or below row m
    pivot_offset = np.argmax(np.abs(A_copy[m:, m]))
    pivot_row = pivot_offset + m

    #Check for near singular pivot
    if np.isclose(A_copy[pivot_row, m], 0.0):
        raise ValueError(f"Matrix is singular or nearly singular at column {m}")

    #Swap rows m and pivot_row if needed
    if pivot_row != m:
        A_copy[[m, pivot_row], :] = A_copy[[pivot_row, m], :]
        v_copy[m], v_copy[pivot_row] = v_copy[pivot_row], v_copy[m]

    return A_copy, v_copy, pivot_row

def random_matrix(N, make_well_conditioned=True):
    """Create a random NxN matrix A and random RHS v.
    If make_well_conditioned is True, add N*I to reduce chance of singularity.
    """
    A = np.random.randn(N, N)
    if make_well_conditioned:
        A += N * np.eye(N)
    v = np.random.randn(N)
    return A.astype(float), v.astype(float)


if __name__ == '__main__':
    #A_in = 1
    #v_in = 1
    #PartialPivot(A_in, v_in)

    #test matrix
    test_matrix = np.array([
        [ 2, 1, 4, 1],
        [ 3, 4, -1, -1],
        [ 1, -4, 1, 5],
        [ 2, -2, 1, 3]
    ], dtype=float)
    test_v = [-4, 3, 9, 7]

    print(GaussElim(test_matrix, test_v))
    print(test_matrix)
    print(PartialPivot(test_matrix, test_v))