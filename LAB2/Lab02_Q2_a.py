import numpy as np
import matplotlib.pyplot as plt
import sympy
import time


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


def exact_int(diff_func, lower_limit, upper_limit):
    # sympy integration for reference
    xs = sympy.Symbol('xs', real=True)  # the variable of integration
    return sympy.integrate(diff_func(xs), (xs, lower_limit, upper_limit))


if __name__ == '__main__':
    a = 0.0  # beginning of interval
    b = 1.0  # end of interval

    # exact integral value
    exact_val = exact_int(f, a, b)

    trap_val = trapezoidal_int(f, N_steps=4, lower_limit=a, upper_limit=b)
    simp_val = simpson_int(f, N_steps=4, lower_limit=a, upper_limit=b)

    print("==========================Question 1B==============================================")
    print("Trapezoidal rule: ", trap_val)
    print("Error (Trapezoidal): ", abs(trap_val - exact_val))

    print("Simpson's rule: ", simp_val)
    print("Error (Simpson): ", abs(simp_val - exact_val))

    print("Exact value: ", exact_val)


    print("==========================Question 1E==============================================")

    print("==========================Question 1C==============================================")
    N_steps_trap = 2**12
    N_steps_simp = 2**4
    trap_val = trapezoidal_int(f, N_steps=N_steps_trap, lower_limit=a, upper_limit=b)
    simp_val = simpson_int(f, N_steps=N_steps_simp, lower_limit=a, upper_limit=b)

    print("Trapezoidal rule: ", trap_val)
    print("Error (Trapezoidal): ", abs(trap_val - exact_val))

    print("Simpson's rule: ", simp_val)
    print("Error (Simpson): ", abs(simp_val - exact_val))

    print("Exact value: ", exact_val)

    print("==========================Question 1D==============================================")

    repeats = 1000

    # time trapezoidal
    start = time.time()
    for _ in range(repeats):
        trapezoidal_int(f, N_steps=N_steps_trap, lower_limit=a, upper_limit=b)
    end = time.time()
    avg_time_trap = (end - start) / repeats

    # time simpson
    start = time.time()
    for _ in range(repeats):
        simpson_int(f, N_steps=N_steps_simp, lower_limit=a, upper_limit=b)
    end = time.time()
    avg_time_simp = (end - start) / repeats

    print(f"Average time Trapezoidal (N={N_steps_trap}): {avg_time_trap:.6e} seconds")
    print(f"Average time Simpson (N={N_steps_simp}):    {avg_time_simp:.6e} seconds")

    print("==========================Question 1D==============================================")
    N1 = 16
    N2 = 32

    T1 = trapezoidal_int(f, N_steps=N1, lower_limit=a, upper_limit=b)
    T2 = trapezoidal_int(f, N_steps=N2, lower_limit=a, upper_limit=b)

    error_est = (T2-T1)/3

    print(f"N1 = {N1:2d}, T1 = {T1:.10f}")
    print(f"N2 = {N2:2d}, T2 = {T2:.10f}")
    print(f"Estimated error (|T2–T1|/3) = {abs(error_est):.10e}")
    print(f"Actual error at N2          = {abs(T2 - exact_val):.10e}")
