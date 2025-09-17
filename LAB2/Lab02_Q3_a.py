
import numpy as np
import matplotlib.pyplot as plt
import sympy

def potential(u, r, z):
    l=1
    Q=1
    eps = 8.854e-12
    numerator = Q*np.exp(-np.tan(u)**2)
    denominator = 4*np.pi*eps*((np.cos(u))**2)*np.sqrt((z-l*np.tan(u))**2 + r**2)
    return numerator/denominator

def potential_integrand(r, z):
    return lambda u: potential(u, r, z)

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

a = -np.pi/2
b = np.pi/2

print("Simpsons rule: ", simpson_int(potential_integrand(1, 0), N_steps=100, lower_limit=a, upper_limit=b))