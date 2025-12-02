import numpy as np
from random import random, randrange, seed
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Ising helper functions
# -----------------------------
def calculate_energy(J_, dipoles):
    horiz = dipoles[:, :-1] * dipoles[:, 1:]    # horizontal neighbours
    vert  = dipoles[:-1, :] * dipoles[1:, :]    # vertical neighbours
    return -J_ * (horiz.sum() + vert.sum())


def run_ising(T, L=20, J=1.0, steps=200000, save_every=1000):
    dipoles = np.random.choice([-1, 1], size=(L, L))
    states = []

    E = calculate_energy(J, dipoles)

    for k in range(steps):
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

        if k % save_every == 0:
            states.append(dipoles.copy())

    return states


# Run simulations at 3 temperatures
Ts = [1.0, 2.0, 3.0]
L = 20
steps = 1000000
save_every = 1000

all_states = [run_ising(T, L=L, steps=steps, save_every=save_every) for T in Ts]


# Create 3 side-by-side animations
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
quivers = []
titles = []

X, Y = np.meshgrid(np.arange(L), np.arange(L))
U = np.zeros((L, L))   # x-component of arrow is always 0

for ax, T, states in zip(axes, Ts, all_states):
    spins0 = states[0]

    V = (-spins0).flatten()
    colors = np.where(spins0 == 1, "red", "blue").flatten()

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

    ax.set_title(f"T = {T}", fontsize=14)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    quivers.append(Q)
    titles.append(ax)

max_frames = min(len(states) for states in all_states)


def update(frame):
    for Q, T, states in zip(quivers, Ts, all_states):
        spins = states[frame]
        V_new = (-spins).flatten()
        colors_new = np.where(spins == 1, "red", "blue").flatten()
        Q.set_UVC(U, V_new)
        Q.set_color(colors_new)

    fig.suptitle(f"Monte Carlo Step {frame * save_every}", fontsize=16)
    return quivers


ani = animation.FuncAnimation(
    fig,
    update,
    frames=max_frames,
    interval=80,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()
