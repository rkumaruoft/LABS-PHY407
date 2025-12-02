"""
Simulated annealing for 2D problems from Newman Exercise 10.10

- Part (i): f(x,y) = x^2 - cos(4*pi*x) + (y-1)^2, start at (2,2).
- Part (ii): f(x,y) = cos(x) + cos(sqrt(2)*x) + cos(sqrt(3)*x) + (y-1)^2,
             domain 0 < x < 50, -20 < y < 20.

Gaussian proposals: dx, dy ~ N(0, sigma).
Exponential cooling: T(t) = Tmax * exp(-t/tau).
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters

seed_global = 8675309
random.seed(seed_global)
np.random.seed(seed_global)

# Annealing schedule
Tmax = 5.0                  # initial temperature
Tmin = 1e-4                 # final temperature
tau = 5000.0                # cooling time constant
max_steps = 50000           # maximum Monte Carlo steps per run

proposal_sigma = 1.0        # std dev for Gaussian proposals
record_every = 1            # record every step


# Objective functions

def f_part_i(x, y):
    return x*x - math.cos(4*math.pi*x) + (y - 1.0)**2

def f_part_ii(x, y):
    return math.cos(x) + math.cos(math.sqrt(2.0)*x) + math.cos(math.sqrt(3.0)*x) + (y - 1.0)**2


def temperature(t, Tmax, tau):
    return Tmax * math.exp(-t / tau)

def metropolis_accept(deltaE, T):
    if deltaE <= 0:
        return True
    if T <= 0:
        return False
    return random.random() < math.exp(-deltaE / T)


def simulated_annealing_2d(f, x0, y0, Tmax, Tmin, tau, max_steps,
                          proposal_sigma=1.0, record_every=1, bounds=None, run_seed=None):
    """
    Simulated annealing for a 2D function f(x,y).
    - bounds: None or ((xmin,xmax),(ymin,ymax)). If provided, proposals outside are rejected.
    - returns dict with history arrays and best/final states.
    """
    if run_seed is not None:
        random.seed(run_seed)
        np.random.seed(run_seed + 407)

    x = float(x0); y = float(y0)
    fx = f(x, y)
    best_x, best_y, best_fx = x, y, fx

    times = []
    temps = []
    xs = []
    ys = []
    fs = []

    t = 0
    while t < max_steps:
        t += 1
        T = temperature(t, Tmax, tau)
        if T < Tmin:
            break

        # propose Gaussian step
        dx = np.random.normal(0.0, proposal_sigma)
        dy = np.random.normal(0.0, proposal_sigma)
        x_new = x + dx
        y_new = y + dy

        # If bounds provided, reject proposals outside domain (do not reflect)
        if bounds is not None:
            (xmin, xmax), (ymin, ymax) = bounds
            if not (xmin < x_new < xmax and ymin < y_new < ymax):
                # record current state and continue (proposal rejected)
                if (t % record_every) == 0:
                    times.append(t); temps.append(T); xs.append(x); ys.append(y); fs.append(fx)
                continue

        fx_new = f(x_new, y_new)
        deltaE = fx_new - fx

        if metropolis_accept(deltaE, T):
            x, y, fx = x_new, y_new, fx_new
            if fx < best_fx:
                best_x, best_y, best_fx = x, y, fx
        # else reject and keep current x,y

        if (t % record_every) == 0:
            times.append(t); temps.append(T); xs.append(x); ys.append(y); fs.append(fx)

    return {
        "times": np.array(times),
        "temps": np.array(temps),
        "xs": np.array(xs),
        "ys": np.array(ys),
        "fs": np.array(fs),
        "best_x": best_x,
        "best_y": best_y,
        "best_f": best_fx,
        "final_x": x,
        "final_y": y,
        "final_f": fx,
        "steps": t
    }


# part (i)

print("=== Part (i): 2D function with global minimum at (0,1) ===")
start_x_i, start_y_i = 2.0, 2.0
res_i = simulated_annealing_2d(
    f_part_i, start_x_i, start_y_i,
    Tmax=Tmax, Tmin=Tmin, tau=tau, max_steps=max_steps,
    proposal_sigma=proposal_sigma, record_every=record_every,
    bounds=None, run_seed=seed_global + 1
)

print(f"Steps: {res_i['steps']}")
print(f"Final (x,y) = ({res_i['final_x']:.8f}, {res_i['final_y']:.8f}), f = {res_i['final_f']:.8f}")
print(f"Best  (x,y) = ({res_i['best_x']:.8f}, {res_i['best_y']:.8f}), f = {res_i['best_f']:.8f}")

# Plot x and y vs time as dots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(res_i['times'], res_i['xs'], '.', markersize=2)
plt.xlabel('MC step'); plt.ylabel('x'); plt.title('Part (i): x vs time (dots)')
plt.subplot(1,2,2)
plt.plot(res_i['times'], res_i['ys'], '.', markersize=2)
plt.xlabel('MC step'); plt.ylabel('y'); plt.title('Part (i): y vs time (dots)')
plt.tight_layout()
plt.show()

# Scatter trajectory in xy-plane
plt.figure(figsize=(5,5))
plt.plot(res_i['xs'], res_i['ys'], '.', markersize=2)
plt.plot(res_i['best_x'], res_i['best_y'], 'r*', markersize=10, label='best')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Part (i): trajectory in (x,y)')
plt.legend(); plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# part (ii)

print("\n=== Part (ii): more complicated x-dependent cosines, domain 0<x<50, -20<y<20 ===")
start_x_ii, start_y_ii = 10.0, 2.0   # you can change starting point
bounds_ii = ((0.0, 50.0), (-20.0, 20.0))

res_ii = simulated_annealing_2d(
    f_part_ii, start_x_ii, start_y_ii,
    Tmax=Tmax, Tmin=Tmin, tau=tau, max_steps=max_steps,
    proposal_sigma=proposal_sigma, record_every=record_every,
    bounds=bounds_ii, run_seed=seed_global + 2
)

print(f"Steps: {res_ii['steps']}")
print(f"Final (x,y) = ({res_ii['final_x']:.8f}, {res_ii['final_y']:.8f}), f = {res_ii['final_f']:.8f}")
print(f"Best  (x,y) = ({res_ii['best_x']:.8f}, {res_ii['best_y']:.8f}), f = {res_ii['best_f']:.8f}")

# Plot x and y vs time as dots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(res_ii['times'], res_ii['xs'], '.', markersize=2)
plt.xlabel('MC step'); plt.ylabel('x'); plt.title('Part (ii): x vs time (dots)')
plt.subplot(1,2,2)
plt.plot(res_ii['times'], res_ii['ys'], '.', markersize=2)
plt.xlabel('MC step'); plt.ylabel('y'); plt.title('Part (ii): y vs time (dots)')
plt.tight_layout()
plt.show()

# Show best found x on a local plot of f(x, y=1) to inspect minima in x
best_x = res_ii['best_x']
xs_plot = np.linspace(max(0, best_x-3.0), min(50, best_x+3.0), 800)
fs_plot = [f_part_ii(x, 1.0) for x in xs_plot]  # slice at y=1
plt.figure(figsize=(6,4))
plt.plot(xs_plot, fs_plot, '-')
plt.axvline(best_x, color='red', linestyle='--', label=f'best x={best_x:.6f}')
plt.xlabel('x'); plt.ylabel('f(x,y=1)'); plt.title('Part (ii): local view around best x (y=1)')
plt.legend(); plt.show()


# multiple restarts for part (ii) to explore competing minima
def multiple_restarts_part_ii(n_runs=8, start_seeds=None):
    if start_seeds is None:
        start_seeds = [seed_global + 100 + i for i in range(n_runs)]
    results = []
    for i, s in enumerate(start_seeds):
        # randomize starting x in (0,50) and y in (-1,3) near expected y=1
        random.seed(s)
        x0 = random.uniform(0, 50)
        y0 = random.uniform(-1, 3)
        r = simulated_annealing_2d(
            f_part_ii, x0, y0, Tmax, Tmin, tau, max_steps,
            proposal_sigma=proposal_sigma, record_every=record_every,
            bounds=bounds_ii, run_seed=s+1000
        )
        results.append((r['best_x'], r['best_y'], r['best_f']))
    return results

# Run a small batch and print summary
batch = multiple_restarts_part_ii(n_runs=12)
print("\nMultiple-restart summary (part ii): best_x, best_y, best_f")
for bx, by, bf in batch:
    print(f"  x={bx:8.5f}  y={by:8.5f}  f={bf:8.5f}")

print("\nFinished.")
