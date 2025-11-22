"""
Lab 08
Question 3: Solving Burgers' Equation
Author: Reeshav Kumar (November 2025)
Purpose: Solve 1D Burgers’ equation u_t + u u_x = 0 on [0, 2π] with given
         Boundary Conditions using leapfrog method.
Outputs: Animation of u(x, t) evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def apply_boundary(u):
    """Apply boundary conditions u(0,t)=0, u(Lx,t)=0"""
    u[0] = 0.0
    u[-1] = 0.0
    return u

if __name__ == "__main__":
    # given constants
    eps = 1
    delta_x = 0.02
    delta_t = 0.005
    Lx = 2 * np.pi
    Tf = 2

    # grids
    x = np.arange(0, Lx + delta_x, delta_x)
    t = np.arange(0, Tf + delta_t, delta_t)
    Nx, Nt = len(x), len(t)

    # arrays
    u_prev = np.sin(x)
    u_curr = np.zeros(Nx)
    u_next = np.zeros(Nx)

    u_prev = apply_boundary(u_prev)
    u_curr = apply_boundary(u_curr)
    u_next = apply_boundary(u_next)

    beta = eps * delta_t / delta_x

    # Forward Euler
    for i in range(1, Nx - 1):
        u_curr[i] = u_prev[i] - 0.5 * beta * ((u_prev[i + 1])**2 - (u_prev[i - 1])**2)
    u_curr = apply_boundary(u_curr)

    # Set up figure for animation
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, u_prev, color='royalblue', lw=2)
    ax.set_xlim(0, Lx)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.grid(True)
    title = ax.text(0.5, 1.02, "Time = 0.000 s", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12)

    frames = []
    times = []

    # leapfrog method for time evolution
    for j in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            u_next[i] = (
                u_prev[i] -
                beta * ((u_curr[i + 1])**2 - (u_curr[i - 1])**2) / 2
            )
        u_next = apply_boundary(u_next)
        u_prev, u_curr = u_curr, u_next

        frames.append(u_curr.copy())
        times.append(j * delta_t)

    # Animation update function
    def update(frame_index):
        y = frames[frame_index].copy()
        y[~np.isfinite(y)] = np.nan
        line.set_ydata(y)
        title.set_text(f"Time = {times[frame_index]:.3f} s")
        return line, title

    # Create and show animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=20, blit=False, repeat=False
    )

    plt.tight_layout()
    plt.xticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )
    plt.show()
