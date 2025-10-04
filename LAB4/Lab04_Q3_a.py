import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    # Part (a):
    c_val = 2.0
    x0 = 0.5
    x_sol, iters = relaxation(c_val, x0, convergence=1e-6, max_iter=100000)
    print("Part (a):")
    print(f"c = {c_val}, solution x = {x_sol:.9f}, iterations = {iters}")

    # Part (b):
    cs = np.arange(0.0, 3, 0.001)
    xs = np.zeros_like(cs)

    xs[0], _ = relaxation(cs[0], 0.0, convergence=1e-8, max_iter=100000)

    for i in range(1, len(cs)):
        c = cs[i]

        # Use previous solution as initial guess
        initial = xs[i - 1]

        # For nonzero branch
        if initial == 0.0:
            initial = 1e-8

        x_sol, iters = relaxation(c, initial, convergence=1e-8, max_iter=200000)
        if x_sol == -1:
            x_sol = 0.0
        xs[i] = x_sol

    plt.figure()
    plt.plot(cs, xs, lw=2)
    plt.axvline(1.0, color='gray', ls='--', label='c = 1 (transition)')
    plt.xlabel('c')
    plt.ylabel('x (solution of x = 1 - e^{-c x})')
    plt.title('Fixed-point relaxation solution x vs c')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()