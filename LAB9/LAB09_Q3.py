"""
Lab 09
Question 3: Solving Burgers' Equation
Author: Reeshav Kumar (November 2025)

Purpose:
    Solve the 1D Burgers’ equation:
        d(u)/dt + e d(u^2 / 2)/dx = 0
    on the interval [0, 2π] using the Lax–Wendroff method.

Outputs:
    PNG plots of u(x,t) at t = 0, 0.5, 1.0, 1.5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def apply_boundary(u):
    """Apply boundary conditions u(0,t)=0, u(L,t)=0."""
    u[0] = 0.0
    u[-1] = 0.0
    return u


if __name__ == "__main__":

    # constants
    eps = 1.0
    dx = 0.02  # space step
    dt = 0.005  # time step
    L = 2 * np.pi  # x range
    Tf = 2.0  # time range

    # define the space and time domain
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, Tf + dt, dt)

    Nx = len(x)
    Nt = len(t)

    # arrays
    u_curr = apply_boundary(np.sin(x))
    u_next = apply_boundary(np.zeros(Nx))

    beta = eps * dt / dx

    # snapshots
    snapshot_times = [0.0, 0.5, 1.0, 1.5]
    snapshots = {ts: None for ts in snapshot_times}
    snapshots[0.0] = u_curr.copy()

    # -------------------------------
    # Lax–Wendroff time stepping
    # -------------------------------
    for j in range(1, Nt):

        for i in range(1, Nx - 1):
            # first-order term
            term2 = (beta / 4) * (u_curr[i + 1] ** 2 - u_curr[i - 1] ** 2)

            # second-order term
            term3 = (beta ** 2 / 4) * (
                    (u_curr[i] + u_curr[i + 1]) * (u_curr[i + 1] ** 2 - u_curr[i] ** 2)
                    -
                    (u_curr[i] + u_curr[i - 1]) * (u_curr[i] ** 2 - u_curr[i - 1] ** 2)
            )

            u_next[i] = u_curr[i] - term2 + term3

        apply_boundary(u_next)

        # swap arrays
        u_curr, u_next = u_next, u_curr
        # save snapshot at required time
        t_now = j * dt
        for ts in snapshot_times:
            if snapshots[ts] is None and abs(t_now - ts) < 0.5 * dt:
                snapshots[ts] = u_curr.copy()

    for tval, profile in snapshots.items():
        plt.figure(figsize=(8, 4))
        plt.plot(x, profile, color='royalblue', linewidth=2)

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title(f"L-W Solution to Burgers' Equation at t = {tval:.1f}")
        plt.grid(True)

        plt.xticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$",
             r"$\frac{3\pi}{2}$", r"$2\pi$"]
        )

        plt.tight_layout()
        plt.savefig(f"LW_burgers_t{tval:.1f}.png", dpi=300)
        plt.close()
