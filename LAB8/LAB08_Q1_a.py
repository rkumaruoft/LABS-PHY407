import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.ma as ma
import pandas as pd


def boundary_T(x, y):
    eps = 1e-12
    # AB
    if abs(y - 0) < eps and 0 <= x <= 5:
        return x
    # BC
    elif abs(x - 5) < eps and 0 <= y <= 3:
        return 5 + (2 / 3) * y
    # CD
    elif abs(y - 3) < eps and 5 <= x <= 15:
        return 7
    # DE
    elif abs(x - 15) < eps and 0 <= y <= 3:
        return 5 + (2 / 3) * y
    # EF
    elif abs(y - 0) < eps and 15 <= x <= 20:
        return 20 - x
    # FG
    elif abs(x - 20) < eps and 0 <= y <= 8:
        return (5 / 4) * y
    # GH
    elif abs(y - 8) < eps and 0 <= x <= 20:
        return 10
    # HA
    elif abs(x - 0) < eps and 0 <= y <= 8:
        return (5 / 4) * y
    else:
        return 0


if __name__ == "__main__":
    # grid spacing
    a = 0.1  # cm

    # define the empty Temp array T(x,y)
    x = np.arange(0, 20 + a, a)
    y = np.arange(0, 8 + a, a)
    cols, rows = len(x), len(y)
    T = np.zeros((rows, cols))

    # Cutout region: 5+dx ≤ x ≤ 15−dx, 0 ≤ y ≤ 3
    eps = 1e-9
    for i in range(cols):
        for j in range(rows):
            if (5 < x[i] < 15) and (0 <= y[j] < 3):
                T[j, i] = -1  # set the value of T in the cutout region -1
            else:
                T[j, i] = boundary_T(x[i], y[j])  # initialize the array with boundary conditions

    # main loop
    target = 1e-6  # celsius
    w = 0.9
    delta = 1.0
    iteration = 0

    # set up plot once
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 4))
    T_masked = ma.masked_where(T == -1, T)
    contour = ax.contourf(X, Y, T_masked, levels=50, cmap="turbo")
    yy, xx = np.where(T != -1)
    ax.scatter(x[xx], y[yy], color='k', s=5, alpha=0.5, label="Grid Points")
    plt.colorbar(contour, label="Temperature (°C)")
    ax.set_title("Steady-State Temperature Distribution")

    # ---- animation update ----
    def update(frame):
        global T, delta, iteration
        iteration += 1
        delta = 0.0
        for j in range(1, rows - 1):
            for i in range(1, cols - 1):
                if T[j, i] == -1 or boundary_T(x[i], y[j]) != 0:
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
        ax.set_title(f"Iteration {iteration}, max ΔT = {delta:.3e}")
        ax.axis("equal")

        # stop animation automatically
        if delta < target:
            ani.event_source.stop()
            print(f"Converged after {iteration} iterations (ΔT={delta:.3e})")

        return []


    ani = animation.FuncAnimation(fig, update, frames=10000, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
