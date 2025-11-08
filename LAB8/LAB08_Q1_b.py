"""
Lab 08
Question 1: Temperature distribution in a heat conductor
Author: Reeshav Kumar (November 2025)
Purpose: Compute the steady-state temperature T(x, y) on a rectangular cut-out
         with specified boundary temperatures.
Outputs: Animated contour plot and saved PNG image showing the temperature field.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.ma as ma
from LAB08_Q1_a import boundary_T, is_boundary_point


if __name__ == "__main__":
    # grid spacing
    a = 0.1  # cm

    # define the empty Temp array T(x,y)
    x = np.arange(0, 20 + a, a)
    y = np.arange(0, 8 + a, a)
    cols, rows = len(x), len(y)
    T = np.zeros((rows, cols))

    # Cutout region: 5 < x < 15, 0 < y < 3
    for i in range(cols):
        for j in range(rows):
            if (5 < x[i] < 15) and (0 <= y[j] < 3):
                T[j, i] = -1  # set the value of T in the cutout region -1
            else:
                T[j, i] = boundary_T(x[i], y[j])  # initialize the array with boundary conditions

    target = 1e-6  # celsius
    w = 0.9        # relaxation factor (change this for different plots)
    delta = 1.0
    iteration = 0

    # set up the plot once
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 4))
    T_masked = ma.masked_where(T == -1, T)
    contour = ax.contourf(X, Y, T_masked, levels=50, cmap="turbo")
    yy, xx = np.where(T != -1)
    ax.scatter(x[xx], y[yy], color='k', s=5, alpha=0.5, label="Grid Points")
    plt.colorbar(contour, label="Temperature (°C)")
    ax.set_title(f"Temperature Distribution (Omega = {w})")

    # ---- animation update ----
    def update(frame):
        global T, delta, iteration
        print(iteration)
        iteration += 1
        delta = 0.0

        for j in range(1, rows - 1):
            for i in range(1, cols - 1):
                if T[j, i] == -1 or is_boundary_point(x[i], y[j]) != 0:
                    continue

                oldT = T[j, i]
                T[j, i] = ((1 + w) * (T[j, i + 1] + T[j, i - 1] + T[j + 1, i] + T[j - 1, i]) / 4) - (w * oldT)
                d = abs(T[j, i] - oldT)
                if d > delta:
                    delta = d

        ax.clear()
        T_masked = ma.masked_where(T == -1, T)
        cp = ax.contourf(X, Y, T_masked, levels=50, cmap="turbo")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_title(f"Temperature Distribution (Omega = {w})")
        ax.axis("equal")

        # stop animation automatically after 100 iterations
        if iteration == 100:
            ani.event_source.stop()

            # save the final temperature plot
            T_masked = ma.masked_where(T == -1, T)
            plt.figure(figsize=(10, 4))
            plt.contourf(X, Y, T_masked, levels=50, cmap="turbo")
            plt.colorbar(label="Temperature (°C)")
            plt.xlabel("x [cm]")
            plt.ylabel("y [cm]")
            plt.title(f"Temperature Distribution (Omega = {w}) after {iteration} iterations")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(f"temperature_distribution_omega_{w}.png", dpi=300)
            plt.close()

        return []

    ani = animation.FuncAnimation(fig, update, frames=10000, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
