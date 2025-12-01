"""
TSP simulated annealing sensitivity study for cooling time constant tau.

- Fix city set with initial seed ns.
- For each tau in taus, run n_runs annealing trials, each with a different run_seed.
- Record final best distances and plot results.

"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
N = 25                   # number of cities
ns = 8675309             # initial seed to generate the fixed city set
Tmax = 10.0              # initial temperature
Tmin = 1e-3              # stopping temperature
max_steps = 200000       # safety cap on steps per run
taus = [1e3, 1e4, 5e4]   # list of tau values to test (shorter -> faster cooling)
n_runs = 20              # number of independent annealing runs per tau (vary second seed)
plot_interval = 0        # set >0 to print progress periodically

def random_cities(n, seed):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.rand(n, 2)

def tour_length(r, order):
    coords = r[order]
    coords = np.vstack([coords, coords[0]])
    diffs = coords[1:] - coords[:-1]
    return np.hypot(diffs[:,0], diffs[:,1]).sum()

def simulated_annealing(r, Tmax, Tmin, tau, max_steps, run_seed, verbose=False):
    """
    Run simulated annealing on fixed city coordinates r.
    Returns: best_distance, best_order, final_order, final_distance
    """
    N = len(r)

    random.seed(run_seed)
    # initial tour: 0..N-1
    order = list(range(N))
    D = tour_length(r, order)
    best_order = order.copy()
    best_D = D

    t = 0
    T = Tmax
    while T > Tmin and t < max_steps:
        t += 1
        T = Tmax * math.exp(-t / tau)

        # choose two distinct indices to swap (allow any)
        i = random.randrange(0, N)
        j = random.randrange(0, N)
        while j == i:
            j = random.randrange(0, N)

        # swap
        order[i], order[j] = order[j], order[i]
        newD = tour_length(r, order)
        deltaD = newD - D

        if deltaD <= 0 or random.random() < math.exp(-deltaD / max(T, 1e-12)):
            D = newD
            if D < best_D:
                best_D = D
                best_order = order.copy()
        else:
            # reject: swap back
            order[i], order[j] = order[j], order[i]

        if plot_interval and (t % plot_interval == 0) and verbose:
            print(f"t={t} T={T:.5g} D={D:.5g} best={best_D:.5g}")

    final_order = order.copy()
    final_D = D
    return best_D, best_order, final_D, final_order

def plot_tour(r, order, title=None, annotate_indices=False):
    coords = r[order]
    coords = np.vstack([coords, coords[0]])
    plt.figure(figsize=(5,5))
    plt.plot(coords[:,0], coords[:,1], '-o', markersize=5)
    if annotate_indices:
        for i,(x,y) in enumerate(r):
            plt.text(x, y, str(i), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.gca().set_aspect('equal', adjustable='box')
    if title:
        plt.title(title)
    plt.show()


# Generate fixed city set

r = random_cities(N, ns)
print(f"Fixed city set generated with seed ns={ns}. Number of cities: {N}")


# Run

results = {}  # tau -> list of best_D values
best_orders_by_tau = {}  # store best order per tau (overall best among runs)
worst_orders_by_tau = {}  # store worst order per tau (overall worst among runs)

for tau in taus:
    print(f"\nRunning experiments for tau = {tau}")
    bests = []
    best_overall = None
    best_overall_D = float('inf')
    worst_overall = None
    worst_overall_D = -float('inf')

    # choose run seeds (different second seeds)
    run_seeds = [ns + 1000 + k for k in range(n_runs)]

    for k, run_seed in enumerate(run_seeds):
        best_D, best_order, final_D, final_order = simulated_annealing(
            r, Tmax, Tmin, tau, max_steps, run_seed, verbose=False
        )
        bests.append(best_D)
        if best_D < best_overall_D:
            best_overall_D = best_D
            best_overall = best_order.copy()
        if best_D > worst_overall_D:
            worst_overall_D = best_D
            worst_overall = best_order.copy()

        if (k+1) % max(1, n_runs//5) == 0:
            print(f"  run {k+1}/{n_runs}: run_seed={run_seed}  best_D={best_D:.6f}")

    results[tau] = np.array(bests)
    best_orders_by_tau[tau] = (best_overall_D, best_overall)
    worst_orders_by_tau[tau] = (worst_overall_D, worst_overall)


# Summarize and plot

# Print summary statistics
print("\nSummary statistics (best D per run):")
for tau in taus:
    arr = results[tau]
    print(f"tau={tau:8g}  n={len(arr)}  mean={arr.mean():.6f}  std={arr.std(ddof=1):.6f}  min={arr.min():.6f}  max={arr.max():.6f}")

# Boxplot of distributions
plt.figure(figsize=(8,5))
data = [results[tau] for tau in taus]
plt.boxplot(data, labels=[str(t) for t in taus], showmeans=True)
plt.xlabel("tau")
plt.ylabel("best tour length D")
plt.title("Distribution of best D across runs for each tau")
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.show()

# Mean vs tau plot (log-x)
means = [results[tau].mean() for tau in taus]
stds = [results[tau].std(ddof=1) for tau in taus]
plt.figure(figsize=(7,5))
plt.errorbar(taus, means, yerr=stds, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel("tau (log scale)")
plt.ylabel("mean best D (error bars = std)")
plt.title("Mean best D vs cooling time constant tau")
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.show()

# Show example tours: best and worst for each tau
for tau in taus:
    bestD, best_order = best_orders_by_tau[tau]
    worstD, worst_order = worst_orders_by_tau[tau]
    print(f"\nExample tours for tau={tau}: best D={bestD:.6f}, worst D={worstD:.6f}")
    plot_tour(r, best_order, title=f"tau={tau} best D={bestD:.6f}")
    plot_tour(r, worst_order, title=f"tau={tau} worst D={worstD:.6f}")

print("\nFinished.")
