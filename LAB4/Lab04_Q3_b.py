import numpy as np


def relaxation(c, initial_guess, convergence=1e-6, max_iter=10000):
    x = initial_guess
    iterations = 0
    while iterations < max_iter:
        x_new = 1 - np.exp(-c * x)
        iterations += 1
        if abs(x_new - x) <= convergence:
            return x_new, iterations
        x = x_new
    return -1, -1


def overrelaxation(c, initial_guess, omega, convergence=1e-6, max_iter=10000):
    x = initial_guess
    iterations = 0
    while iterations < max_iter:
        x_new = ((1 + omega) * (1 - np.exp(-c * x))) - (omega * x)
        iterations += 1
        if abs(x_new - x) <= convergence:
            return x_new, iterations
        x = x_new
        print(f"Current value {x} in {iterations} iterations.")
    return -1, -1


if __name__ == "__main__":
    value, iterations = relaxation(c=2, initial_guess=1)
    print("-------------------Simple Relaxation---------------------")
    print(f"Value converges to {value} in {iterations} iterations.\n")

    omega = 0.5
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")
