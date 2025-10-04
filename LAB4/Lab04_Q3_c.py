import numpy as np
from scipy import constants


def weins_non_linear(x):
    # Nonlinear equation from Wien's displacement law
    return 5 * np.exp(-x) + x - 5


def binary_search(function, x1, x2, convergence=10e-6, max_iter=10000):
    f = function
    f1, f2 = f(x1), f(x2)

    # Check sign change in interval
    if f1 * f2 > 0:
        print("f(x1) and f(x2) must have opposite signs!")
        return -1, -1

    for iteration in range(max_iter):
        x_mid = 0.5 * (x1 + x2)
        f_mid = f(x_mid)

        # Stop if interval small enough or exact root found
        if abs(x2 - x1) <= convergence or f_mid == 0:
            return x_mid, iteration

        # Narrow down interval
        if f1 * f_mid < 0:
            x2, f2 = x_mid, f_mid
        else:
            x1, f1 = x_mid, f_mid

    print(f"Binary search did not converge in {max_iter} iterations")
    return -1, -1


if __name__ == "__main__":
    value, iterations = binary_search(weins_non_linear, 4, 5)
    print(f"-------------------Binary Search Interval[{4, 5}]---------------------")
    print(f"Value converges to {value} in {iterations} iterations.\n")

    print(f"-------------------Temperature of the sun---------------------")
    wavelength = 502e-9  # peak wavelength in meters
    weins_const = constants.h * constants.c / (constants.k * value)
    temp = weins_const / wavelength

    print(str(temp) + " K")
