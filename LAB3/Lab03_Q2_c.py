from common_funcs import *
import scipy.constants as consts
from Lab03_Q2_a import period_integrand
import os

if __name__ == "__main__":

    mass = 1.0  # kg
    spring_const = 12.0  # N/m
    x_c = consts.c * sqrt(mass / spring_const)

    boring_val = 2 * np.pi * np.sqrt(mass / spring_const)

    N_points = 4000
    x0_range = np.linspace(1, 10*x_c, N_points)

    N_quad = 200
    periods = []
    for x0 in x0_range:
        periods.append(gaussian_quad(period_integrand(x0, mass, spring_const),
                              N_quad, 0, x0))

    relative_limit_curve = 4 * x0_range / consts.c

    plt.figure(figsize=(8, 5))

    plt.plot(x0_range, periods, marker=".", linestyle="None", markersize=1,
             label="Gaussian Quad")
    # classical limit
    plt.axhline(y=boring_val, color="r", linestyle="--", lw=2,
                label=r"Classical Limit $T = 2\pi\sqrt{m/k}$")

    # Relativistic limit
    plt.plot(x0_range, relative_limit_curve, "g--", lw=2, label=r"Relativistic Limit $4x_0/c$")

    plt.xlabel(r"Amplitude $x_0$ (m)")
    plt.ylabel("Period (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/period_vs_x0.png", dpi=300, bbox_inches="tight")
    plt.show()
