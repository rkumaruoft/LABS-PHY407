import numpy as np
import matplotlib.pyplot as plt
import sympy

def f(x):
    return 4/(1 + x**2)

def trapezoidal_int(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps  # width of slice
    s = 0.5 * diff_func(lower_limit) + 0.5 * diff_func(upper_limit)  # the end bits
    for k in range(1, N_steps):  # adding the interior bits
        s += f(lower_limit + k * h)
    return h*s

def simpson_int(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps  # width of slice

    sum1 = 0
    for k in range(1, N_steps, 2):
        sum1 += diff_func(lower_limit + k * h)

    sum2 = 0
    for k in range(2, N_steps, 2):
        sum2 += diff_func(lower_limit + k * h)

    s = (h/3)*(diff_func(lower_limit) + diff_func(upper_limit) +
               4*sum1 + 2*sum2)
    return s

def exact_int(diff_func, lower_limit, upper_limit):
    # sympy integration for reference
    xs = sympy.Symbol('xs', real=True)  # the variable of integration
    return sympy.integrate(diff_func(xs), (xs, lower_limit, upper_limit))


a = 0.0 # beginning of interval
b = 1.0 # end of interval

print("Trapezoidal rule: ", trapezoidal_int(f, N_steps = 1000, lower_limit = a, upper_limit = b))
print("Simpsons rule: ", simpson_int(f, N_steps = 1000, lower_limit = a, upper_limit = b))

print("Exact value: ", exact_int(f, a, b))
