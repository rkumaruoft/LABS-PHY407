import numpy as np
from random import random, randrange, seed
import matplotlib.pyplot as plt
import os


def calculate_energy(J_, dipoles):
    # horizontal neighbors: (i, j) with (i, j+1)
    horiz = dipoles[:, :-1] * dipoles[:, 1:]

    # vertical neighbors: (i, j) with (i+1, j)
    vert = dipoles[:-1, :] * dipoles[1:, :]

    # total energy
    return -J_ * (horiz.sum() + vert.sum())


if __name__ == "__main__":
    os.makedirs("q4_plots", exist_ok=True)

    for T in [1.0, 2.0, 3.0]:
        # define constants
        kB = 1.0
        J = 1.0
        L = 20   # 20×20 grid
        N = 1000000  # number of Monte Carlo steps

        # generate 2D array of dipoles (spins)
        dipoles = np.random.choice([-1, 1], size=(L, L))

        # diagnostics
        energy = []
        magnet = []

        E = calculate_energy(J, dipoles)
        energy.append(E)
        magnet.append(np.sum(dipoles))

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

        # Energy Plot
        plt.figure(figsize=(8, 4))
        plt.plot(energy)
        plt.xlabel("Monte Carlo step (t)")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Monte Carlo steps (T = {T})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"q4_plots/energy_islingT={T}.png", dpi=300)
        plt.clf()

        # magnetization plot
        plt.figure(figsize=(8, 4))
        plt.plot(magnet)
        plt.xlabel("Monte Carlo step (t)")
        plt.ylabel("Magnetization")
        plt.title(f"Magnetization vs Monte Carlo steps (T = {T})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"q4_plots/magnet_isling_T={T}.png", dpi=300)
        plt.clf()

        print(f"Done for {T}")
