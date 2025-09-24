import numpy as np
import matplotlib.pyplot as plt
import sympy
import time
from pylab import *
from common_funcs import *

def f(x):
    return 4 / (1 + x ** 2)


if __name__ == '__main__':
    #defining variables
    lower_limit = 0
    upper_limit = 1
    N_steps = 8
    diff_func = f

    print(trapezoidal_int(diff_func, N_steps, lower_limit, upper_limit))
    print(simpson_int(diff_func, N_steps, lower_limit, upper_limit))
    print(gaussian_quad(diff_func, N_steps, lower_limit, upper_limit))
