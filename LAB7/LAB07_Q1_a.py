# Solution to Newman 8.8, Space garbage.
# Author: Nico Grisouard, Univ. of Toronto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def rhs(r):
    """ The right-hand-side of the equations
    INPUT:
    r = [x, vx, y, vy] are floats (not arrays)
    note: no explicit dependence on time
    OUTPUT:
    1x2 numpy array, rhs[0] is for x, rhs[1] is for vx, etc"""
    M = 10.
    L = 2.

    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]

    r2 = x ** 2 + y ** 2
    Fx, Fy = - M * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25 * L ** 2))
    return np.array([vx, Fx, vy, Fy], float)


ftsz = 12
font = {'size': ftsz}  # font size
rc('font', **font)


# RK4 single step function (keeps same signature style)
def rk4_step(r, h, deriv):
    k1 = deriv(r)
    k2 = deriv(r + 0.5 * h * k1)
    k3 = deriv(r + 0.5 * h * k2)
    k4 = deriv(r + h * k3)
    return r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Adaptive RK4 using one full vs two half steps
def adaptive_rk4(r0, t0, tmax, h0, delta, deriv,
                 h_min=1e-9, h_max=0.1, p=4, safety=0.9, fac_min=0.2, fac_max=5.0):
    """
    Adaptive RK4 integrator.
    - r0: initial state [x, vx, y, vy]
    - t0: start time
    - tmax: end time
    - h0: initial step size
    - delta: target positional error (scalar)
    Returns: times, traj (array of states), h_values (step sizes at accepted steps)
    """
    t = t0
    r = r0.copy()
    h = h0

    times = [t]
    traj = [r.copy()]
    h_vals = [h]

    while t < tmax:
        if t + h > tmax:
            h = tmax - t

        # one full RK4 step
        r_full = rk4_step(r, h, deriv)

        # two half RK4 steps
        r_half = rk4_step(r, h / 2.0, deriv)
        r_half = rk4_step(r_half, h / 2.0, deriv)

        # position error estimate (epsilon_x, epsilon_y)
        eps_x = r_half[0] - r_full[0]
        eps_y = r_half[2] - r_full[2]
        err_pos = np.sqrt(eps_x ** 2 + eps_y ** 2)

        # compute rho using given formula; guard division by zero
        if err_pos == 0.0:
            rho = np.inf
        else:
            rho = (h * delta) / err_pos

        # accept or reject step
        if rho >= 1.0:
            # accept; use Richardson extrapolation to improve accuracy
            r = r_half + (r_half - r_full) / (2.0 ** p - 1.0)
            t += h
            times.append(t)
            traj.append(r.copy())
            h_vals.append(h)

            # update h
            if np.isfinite(rho):
                h_new = h * min(fac_max, max(fac_min, safety * rho ** (1.0 / (p + 1))))
            else:
                h_new = h * fac_max
            h = min(h_max, max(h_min, h_new))
        else:
            # reject and reduce h
            h = max(h_min, h * max(fac_min, safety * rho ** (1.0 / (p + 1))))

    return np.array(times), np.array(traj), np.array(h_vals)


# Fixed-step RK4 for comparison
def fixed_rk4(r0, t0, tmax, h, deriv):
    t = t0
    r = r0.copy()
    times = [t]
    traj = [r.copy()]
    N = int(np.ceil((tmax - t0) / h))
    for i in range(N):
        if t + h > tmax:
            h = tmax - t
        r = rk4_step(r, h, deriv)
        t += h
        times.append(t)
        traj.append(r.copy())
    return np.array(times), np.array(traj)


if __name__ == "__main__":

    # %% This next part adapted from Newman's odesim.py --------------------------|
    a = 0.0
    b = 10.0
    N = 1000  # let's leave it at that for now
    h = (b - a) / N

    tpoints = np.arange(a, b, h)
    xpoints = []
    vxpoints = []  # the future dx/dt
    ypoints = []
    vypoints = []  # the future dy/dt

    # below: ordering is x, dx/dt, y, dy/dt
    r = np.array([1., 0., 0., 1.], float)
    for t in tpoints:
        xpoints.append(r[0])
        vxpoints.append(r[1])
        ypoints.append(r[2])
        vypoints.append(r[3])
        k1 = h * rhs(r)  # all the k's are vectors
        k2 = h * rhs(r + 0.5 * k1)  # note: no explicit dependence on time of the RHSs
        k3 = h * rhs(r + 0.5 * k2)
        k4 = h * rhs(r + k3)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # run adaptive integrator and fixed-step comparison
    # [x, vx, y, vy]
    r0 = np.array([1., 0., 0., 1.], float)

    # adaptive parameters
    h0 = 0.01
    delta = 1e-6  # target error

    times_ad, traj_ad, h_vals = adaptive_rk4(r0, a, b, h0, delta, rhs)

    # non-adaptive run with h = 0.001 (N = 10000 if b-a = 10)
    h_fixed = 0.001
    times_fixed, traj_fixed = fixed_rk4(r0, a, b, h_fixed, rhs)

    # Prepare plotting arrays
    x_ad = traj_ad[:, 0]
    y_ad = traj_ad[:, 2]
    x_fix = traj_fixed[:, 0]
    y_fix = traj_fixed[:, 2]

    # Trajectory only (adaptive points and fixed-step curve)
    plt.figure(1)
    plt.plot(x_ad, y_ad, 'k.', markersize=3, label='adaptive (accepted points)')
    plt.plot(x_fix, y_fix, 'r-', linewidth=1, alpha=0.7, label=f'fixed h={h_fixed}')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    print('Trajectory of a ball bearing around a space rod.')
    plt.axis('equal')
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=300)
    plt.show()

    # Adaptive h vs time only
    plt.figure(2)
    plt.plot(times_ad, h_vals, 'b.-')
    plt.xlabel('t')
    plt.ylabel('h (adaptive)')
    print('Adaptive step size over time')
    plt.grid()
    plt.tight_layout()
    plt.savefig('adaptive_h.png', dpi=300)
    plt.show()

