import numpy as np
import matplotlib.pyplot as plt
import sympy
import time
from pylab import *
from common_funcs import *
import math

def f(x):
    return 4 / (1 + x ** 2)


def compute_errors(int_method):
    rel_errors = []
    est_rel_errors = []
    for N in Ns:
        I_N = int_method(diff_func, N, lower_limit, upper_limit)
        I_2N = int_method(diff_func, 2 * N, lower_limit, upper_limit)
        true_rel = abs(I_N - I_true) / abs(I_true)
        est_rel = abs(I_2N - I_N) / abs(I_true)
        rel_errors.append(true_rel)
        est_rel_errors.append(est_rel)
    return rel_errors, est_rel_errors

if __name__ == '__main__':

    lower_limit = 0.0
    upper_limit = 1.0
    diff_func = lambda x: 4.0 / (1.0 + x ** 2)
    I_true = float(exact_int(diff_func, lower_limit, upper_limit))

    Ns = [2 ** k for k in range(3, 11)] #8 to 2048

    trap_err, trap_est = compute_errors(trapezoidal_int)
    simp_err, simp_est = compute_errors(simpson_int)
    gauss_err, gauss_est = compute_errors(gaussian_quad)

    #plotting
    plt.figure(figsize=(8, 5))
    plt.loglog(Ns, trap_err, 'o-', label='Trapezoid true rel error')
    plt.loglog(Ns, trap_est, 'x--', label='Trapezoid estimate |I2N - IN|/|I_true|')
    plt.loglog(Ns, simp_err, 's-', label='Simpson true rel error')
    plt.loglog(Ns, simp_est, 'd--', label='Simpson estimate')
    plt.loglog(Ns, gauss_err, '^-', label='Gaussian true rel error')
    plt.loglog(Ns, gauss_est, 'v--', label='Gaussian estimate')
    plt.xlabel('N (number of Gaussian nodes)')
    plt.ylabel('Relative error')
    plt.title('Log-log: relative error vs N')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.show()


