import numpy as np
import matplotlib.pyplot as plt
import sympy
import time
from pylab import *


def f(x):
    return 4 / (1 + x ** 2)


def trapezoidal_int(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps  # width of slice
    s = 0.5 * diff_func(lower_limit) + 0.5 * diff_func(upper_limit)  # the end bits
    for k in range(1, N_steps):  # adding the interior bits
        s += f(lower_limit + k * h)
    return h * s


def simpson_int(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps  # width of slice

    sum1 = 0
    for k in range(1, N_steps, 2):
        sum1 += diff_func(lower_limit + k * h)

    sum2 = 0
    for k in range(2, N_steps, 2):
        sum2 += diff_func(lower_limit + k * h)

    s = (h / 3) * (diff_func(lower_limit) + diff_func(upper_limit) +
                   4 * sum1 + 2 * sum2)
    return s

 def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*np.tan(a)))
    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = ones(N, float)
        p1 = copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))
    # Calculate the weights
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w

def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

def gaussian_quad(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps
    # define N
    N = 10
    # call gausswx for xi, wi
    x, w = gaussxw(N)
    # initialize integral to 0.
    I = 0.
    # loop over sample points to compute integral
    for k in range(N):
        I += w[k] * f(x[k])
    # print
    print(I)
