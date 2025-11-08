"""
Lab 08
Question 3: Solving Burgers' Equation
Author: Reeshav Kumar (November 2025)
Purpose: Solve 1D Burgers’ equation u_t + u u_x = 0 on [0, 2π] with given
         Boundary Conditions using leapfrog method.
Outputs: PNG plots of u(x,t) at t = 0, 0.5, 1.0, 1.5.
"""
import numpy as np
import matplotlib.pyplot as plt

def apply_boundary(u):
    """Apply boundary conditions: u(0,t)=0, u(Lx,t)=0."""
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

    x = np.arange(0, Lx + delta_x, delta_x)  # spatial positions
    t = np.arange(0, Tf + delta_t, delta_t)  # time steps
    Nx = len(x)
    Nt = len(t)

    # define the solution arrays
    u_prev = np.zeros(Nx)  # u(x, t_{j-1})
    u_curr = np.zeros(Nx)  # u(x, t_j)
    u_next = np.zeros(Nx)  # u(x, t_{j+1})

    # initial condition
    u_prev = np.sin(x)

    # apply boundary conditions
    u_prev = apply_boundary(u_prev)
    u_curr = apply_boundary(u_curr)
    u_next = apply_boundary(u_next)

    beta = eps * delta_t / delta_x

    # forward euler step
    for i in range(1, Nx - 1):
        u_curr[i] = (
                u_prev[i]- 0.5 * beta * ((u_prev[i + 1]) ** 2 - (u_prev[i - 1]) ** 2)
        )

    u_curr = apply_boundary(u_curr)

    # times to save and plot
    save_times = [0, 0.5, 1.0, 1.5]
    save_indices = [int(ti / delta_t) for ti in save_times]
    saved_profiles = {0: np.sin(x)}  # initial condition at t=0

    # leapfrog method for time evolution
    for j in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            u_next[i] = (
                    u_prev[i]
                    - 0.5 * beta * ((u_curr[i + 1]) ** 2 - (u_curr[i - 1]) ** 2)
            ) #
        u_next = apply_boundary(u_next)
        u_prev, u_curr = u_curr, u_next

        # Save at selected times
        if j in save_indices:
            saved_profiles[round(j * delta_t, 2)] = u_curr.copy()
            print(f"Saved snapshot at t = {j * delta_t:.2f}")


    # plot results
    for tval, profile in saved_profiles.items():
        plt.figure(figsize=(8, 4))
        plt.plot(x, profile, color='royalblue')
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title(f"Solution to the Burger's Eq. at t = {tval:.1f}")
        plt.grid(True)
        plt.xticks(
            [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
            [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        )

        plt.tight_layout()
        plt.savefig(f"burgers_t{tval:.1f}.png", dpi=300)
        plt.close()





