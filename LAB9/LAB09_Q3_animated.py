"""
Burgers' Animation (Lax–Wendroff)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def apply_boundary(u):
    u[0] = 0.0
    u[-1] = 0.0
    return u


if __name__ == "__main__":

    # constants
    eps = 1.0
    dx = 0.02
    dt = 0.005
    L = 2 * np.pi
    Tf = 2.0 # change this time for longer behaviour

    # grid
    x = np.arange(0, L + dx, dx)
    t_steps = int(Tf / dt)
    Nx = len(x)

    # solution arrays
    u_curr = apply_boundary(np.sin(x))
    u_next = apply_boundary(np.zeros(Nx))

    beta = eps * dt / dx

    # ----- Precompute all frames (fast) -----
    frames = [u_curr.copy()]

    for _ in range(t_steps):
        for i in range(1, Nx - 1):

            term2 = (beta / 4) * (u_curr[i+1]**2 - u_curr[i-1]**2)
            term3 = (beta**2 / 4) * (
                (u_curr[i] + u_curr[i+1]) * (u_curr[i+1]**2 - u_curr[i]**2)
                -
                (u_curr[i] + u_curr[i-1]) * (u_curr[i]**2 - u_curr[i-1]**2)
            )

            u_next[i] = u_curr[i] - term2 + term3

        apply_boundary(u_next)
        u_curr, u_next = u_next, u_curr
        frames.append(u_curr.copy())

    # ----- Animation -----
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], lw=2, color="royalblue")
    time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(np.min(frames), np.max(frames))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Burgers' Equation (Lax–Wendroff)")
    ax.grid(True)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(i):
        line.set_data(x, frames[i])
        time_text.set_text(f"t = {i*dt:.3f}")
        return line, time_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        init_func=init,
        interval=30,
        blit=True,
    )

    plt.show()
