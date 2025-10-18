import numpy as np
import matplotlib.pyplot as plt
import os


def static_friction(velocity, threshold_velocity, beta):
    return -beta * np.exp(-np.abs(velocity) / threshold_velocity)


def fluid_friction(velocity, alpha):
    return -alpha * velocity


if __name__ == "__main__":
    # constants
    beta = 5              # pseudo-static friction coefficient
    alpha = 0.1 * beta    # viscous coefficient
    vf = 5                # characteristic velocity

    # velocity range
    v = np.linspace(0, 8, 3000)
    F_abs = np.abs(fluid_friction(v, alpha) + static_friction(v, vf, beta))

    # ensure the folder exists
    os.makedirs("plots", exist_ok=True)

    # plot
    plt.plot(v, F_abs, linewidth=2)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Total friction (|F|) (N)")
    plt.grid(True)

    # save to file
    plt.savefig("plots/total_friction.png", dpi=300, bbox_inches="tight")

    # show
    plt.show()
