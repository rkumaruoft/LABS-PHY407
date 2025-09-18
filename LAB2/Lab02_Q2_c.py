import os

import numpy as np
from Lab02_Q2_a import simpson_int
from Lab02_Q2_b import bessel_integrand
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N_steps = 1000
    # z at m=3, n=2 ≈ 11.620
    z = 11.620
    bessel_order = 2
    Nr = 201
    N_theta = 200

    # radial and angular grids
    r_over_R = np.linspace(0.0, 1.0, Nr)
    theta_vals = np.linspace(0, 2 * np.pi, N_theta)

    # --- precompute Bessel values for each r/R ---
    bessel_vals = np.array([
        simpson_int(bessel_integrand(z * rR, bessel_order),
                    N_steps, 0, np.pi)
        for rR in r_over_R
    ])

    # --- build 2D grid r/R and theta---
    R, Theta = np.meshgrid(r_over_R, theta_vals)

    # Create a Jn grid and Umn on the grid
    Jn_grid = np.tile(bessel_vals, (N_theta, 1))
    U = Jn_grid * np.cos(bessel_order * Theta)

    # convert to Cartesian coords
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = U

    # --- plotting ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u_mn")
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # multiple views
    for elev, azim in [(30, 45), (60, 120), (20, 200)]:
        ax.view_init(elev=elev, azim=azim)
        plt.show()
    views = [
        (30, 45, "view1.png"),
        (60, 120, "view2.png"),
        (20, 200, "view3.png"),
        (90, 0, "top_down.png"),
    ]

    for elev, azim, name in views:
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(os.path.join("Plots", name), dpi=300, bbox_inches="tight")
