"""Gaussian elimination, partial pivoting,
and LU decomposition approaches"""

"""
For each of a range of values of N, creates a random matrix A and a random array v.

Finds the solution x for the same A and v for the three different methods (Gaussian
elimination, partial pivoting, LU decomposition).

Measures the time it takes to solve for x using each method.

For each case, checks the answer by comparing v_sol = dot(A, x) to the original input array v, using numpy.dot. The check can be carried out by calculating
the mean of the absolute value of the differences between the arrays, which can be
written err = mean(abs(v-v_sol)).

Stores the timings and errors and plots them for each method.
"""

from SolveLinear import *


