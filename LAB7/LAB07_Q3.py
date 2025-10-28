import numpy as np
import matplotlib.pyplot as plt


def f(r):
    """
    Returns derivatives [dx/dt, dy/dt] for:
        dx/dt = 1 - 4x + x^2 y
        dy/dt = 3x - x^2 y
    """
    x, y = r
    return np.array([1 - 4*x + x*x*y, 3*x - x*x*y], dtype=float)


def modified_midpoint(f, r0, h, n):
    """
    Integrate one step of size h using n midpoint substeps.
    """
    h_n = h / n                       # substep size
    r_prev = np.array(r0)
    r_curr = r_prev + h_n * f(r_prev)
    for _ in range(2, n + 1):
        r_next = r_prev + 2*h_n * f(r_curr)
        r_prev, r_curr = r_curr, r_next

    # final correction (averages last two points)
    return 0.5 * (r_curr + r_prev + h_n * f(r_curr))


def bulirsch_stoer(f, r0, t0, t1, delta, nmax=8):
    """
    Adaptive Bulirsch–Stoer step using Richardson extrapolation.
    Refines step until error < delta or splits interval recursively.
    """
    H = t1 - t0                       # total step size
    R_prev = np.empty((1, 2))
    R_prev[0] = modified_midpoint(f, r0, H, 1)  # first row R_11

    # try successive extrapolation rows
    for n in range(2, nmax + 1):
        R_curr = np.empty((n, 2))
        R_curr[0] = modified_midpoint(f, r0, H, n)

        # build Richardson extrapolation table
        for m_idx in range(1, n):
            m = m_idx + 1
            denom = (n / (n - 1)) ** (2 * (m - 1)) - 1
            eps = (R_curr[m_idx - 1] - R_prev[m_idx - 1]) / denom
            R_curr[m_idx] = R_curr[m_idx - 1] + eps
            last_eps = eps            # last correction gives error

        err = np.max(np.abs(last_eps))
        if err <= H * delta:
            return [t1], [R_curr[-1].copy()]

        # otherwise refine further
        R_prev = R_curr

    # if still too large error, split step and recurse
    t_half = t0 + 0.5 * H
    t_left, r_left = bulirsch_stoer(f, r0, t0, t_half, delta, nmax)
    r_mid = r_left[-1]
    t_right, r_right = bulirsch_stoer(f, r_mid, t_half, t1, delta, nmax)
    return t_left + t_right, r_left + r_right


if __name__ == "__main__":
    r0 = np.array([0.0, 0.0])        # initial [x, y]
    a, b = 0.0, 20.0                 # integration limits
    delta = 1e-10                    # target local accuracy

    # integrate system
    t_pts, r_pts = bulirsch_stoer(f, r0, a, b, delta)

    # append initial condition for plotting
    t_pts = np.array([a] + t_pts)
    r_pts = np.array([r0] + r_pts)
    x, y = r_pts[:, 0], r_pts[:, 1]

    # plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(t_pts, x, s=0.2, alpha=0.4, color='blue', label="x(t)")
    plt.scatter(t_pts, y, s=0.2, alpha=0.4, color='green', label="y(t)")
    plt.xlabel("Time (t)")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("concentration.png", dpi=300, bbox_inches="tight")
    plt.show()

    # zoomed-in plot (0 <= t <= 2)
    plt.figure(figsize=(8, 4))
    plt.scatter(t_pts, x, s=0.2, color='blue', alpha=0.8, label="x(t)")
    plt.scatter(t_pts, y, s=0.2, color='green', alpha=0.6, label="y(t)")
    plt.xlim(6, 9)
    plt.xlabel("Time (t)")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("concentration_zoomed.png", dpi=300, bbox_inches="tight")
    plt.show()