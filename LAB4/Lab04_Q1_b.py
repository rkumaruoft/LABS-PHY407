"""Gaussian elimination, partial pivoting,
and LU decomposition approaches"""

import matplotlib.pyplot as plt
from SolveLinear import *

def run_experiment(N_values, trials=3, seed=8675309):
    rng = np.random.default_rng(seed)
    timings = {
        'GaussElim': np.zeros((len(N_values), trials)),
        'GaussElimPP': np.zeros((len(N_values), trials)),
        'LU': np.zeros((len(N_values), trials))
    }
    errors = {
        'GaussElim': np.zeros((len(N_values), trials)),
        'GaussElimPP': np.zeros((len(N_values), trials)),
        'LU': np.zeros((len(N_values), trials))
    }

    for idx, N in enumerate(N_values):
        for t in range(trials):
            #Create random matrix and vector
            A = rng.standard_normal((N, N))
            #Make matrices reasonably well-conditioned by adding identity scaled small factor
            A += np.eye(N) * 1e-3
            v = rng.standard_normal(N)

            #GaussElim (no pivoting)
            start = time.perf_counter()
            try:
                x_ge = GaussElim(A, v)
            except Exception as e:
                x_ge = np.full_like(v, np.nan)
            elapsed = time.perf_counter() - start
            timings['GaussElim'][idx, t] = elapsed
            if not np.any(np.isnan(x_ge)):
                v_sol = A.dot(x_ge)
                errors['GaussElim'][idx, t] = np.mean(np.abs(v - v_sol))
            else:
                errors['GaussElim'][idx, t] = np.nan

            #GaussElimPP (partial pivoting)
            start = time.perf_counter()
            try:
                x_pp = GaussElimPP(A, v)
            except Exception as e:
                x_pp = np.full_like(v, np.nan)
            elapsed = time.perf_counter() - start
            timings['GaussElimPP'][idx, t] = elapsed
            if not np.any(np.isnan(x_pp)):
                v_sol = A.dot(x_pp)
                errors['GaussElimPP'][idx, t] = np.mean(np.abs(v - v_sol))
            else:
                errors['GaussElimPP'][idx, t] = np.nan

            #LU-based solver
            start = time.perf_counter()
            try:
                x_lu = solve_with_lu(A, v)
            except Exception as e:
                x_lu = np.full_like(v, np.nan)
            elapsed = time.perf_counter() - start
            timings['LU'][idx, t] = elapsed
            if not np.any(np.isnan(x_lu)):
                v_sol = A.dot(x_lu)
                errors['LU'][idx, t] = np.mean(np.abs(v - v_sol))
            else:
                errors['LU'][idx, t] = np.nan

    #mean across trials
    timings_mean = {k: np.nanmean(v, axis=1) for k, v in timings.items()}
    errors_mean = {k: np.nanmean(v, axis=1) for k, v in errors.items()}
    return timings_mean, errors_mean


def plot_results(N_values, timings_mean, errors_mean):
    # Timing plot
    plt.figure()
    plt.plot(N_values, timings_mean['GaussElim'], marker='o', label='GaussElim (no pivot)')
    plt.plot(N_values, timings_mean['GaussElimPP'], marker='s', label='GaussElimPP (partial pivot)')
    plt.plot(N_values, timings_mean['LU'], marker='^', label='LU (scipy)')
    plt.xlabel('Matrix size N')
    plt.ylabel('Time (s)')
    plt.title('Solve time vs N')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # Error plot
    plt.figure()
    plt.plot(N_values, errors_mean['GaussElim'], marker='o', label='GaussElim (no pivot)')
    plt.plot(N_values, errors_mean['GaussElimPP'], marker='s', label='GaussElimPP (partial pivot)')
    plt.plot(N_values, errors_mean['LU'], marker='^', label='LU (scipy)')
    plt.xlabel('Matrix size N')
    plt.ylabel('Mean absolute residual error')
    plt.title('Accuracy (mean |v - A x|) vs N')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    N_values = np.arange(5, 300, 5)
    timings_mean, errors_mean = run_experiment(N_values, trials=3, seed=8675309)

    plot_results(N_values, timings_mean, errors_mean)

