import numpy as np
from random import random, randrange, seed
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation


def calculate_energy(J_, dipoles):
    # horizontal neighbors: (i, j) with (i, j+1)
    horiz = dipoles[:, :-1] * dipoles[:, 1:]

    # vertical neighbors: (i, j) with (i+1, j)
    vert = dipoles[:-1, :] * dipoles[1:, :]

    # total energy
    return -J_ * (horiz.sum() + vert.sum())


if __name__ == "__main__":
    seed(1234)
    # define constants
    kB = 1.0
    T = 3.0  # change this value for different animations
    J = 1.0
    L = 20  # 20×20 grid
    N = 1000000  # number of Monte Carlo steps

    # generate 2D array of dipoles (spins)
    dipoles = np.random.choice([-1, 1], size=(L, L))

    energy = []
    magnet = []

    # for animation
    states = []  # store snapshots
    save_every = 1000

    E = calculate_energy(J, dipoles)
    energy.append(E)
    magnet.append(np.sum(dipoles))

    # starting state
    states.append(dipoles.copy())

    for k in range(N):
        i = randrange(L)
        j = randrange(L)

        dipoles[i, j] *= -1
        E_new = calculate_energy(J, dipoles)

        if E_new <= E:
            E = E_new
        else:
            dE = E_new - E
            if random() < np.exp(-dE / T):
                E = E_new
            else:
                dipoles[i, j] *= -1

        energy.append(E)
        magnet.append(np.sum(dipoles))

        if k % save_every == 0:
            states.append(dipoles.copy())

    fig, ax = plt.subplots(figsize=(6, 6))
    L = states[0].shape[0]

    # Grid of arrow positions
    X, Y = np.meshgrid(np.arange(L), np.arange(L))

    # Initial spins
    spins0 = states[0]

    # Set up flattened values for quiver
    U = np.zeros((L, L))
    V = (-spins0).flatten()  # y-component: +1 up, -1 down
    colors = np.where(spins0 == 1, "red", "blue").flatten()

    # Create quiver object
    Q = ax.quiver(
        X.flatten() + 0.5,
        Y.flatten() + 0.5,
        U,
        V,
        color=colors,
        pivot="mid",
        scale=30,
        headwidth=4,
        headlength=6,
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    title = ax.text(0.5, 1.05, f"T = {T}, Step 0", transform=ax.transAxes,
                    ha="center", fontsize=16)


    def update(frame):
        spins = states[frame]
        V_new = (-spins).flatten()
        colors_new = np.where(spins == 1, "red", "blue").flatten()

        Q.set_UVC(U, V_new)  # update arrow directions
        Q.set_color(colors_new)  # update arrow colors

        title.set_text(f"T = {T}, Step {frame * save_every}")
        return [Q]


    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=70,
        blit=False,
        repeat=False
    )

    plt.show()
