from random import random, randrange, seed
import numpy as np
import matplotlib.pyplot as plt
import os


# Monte Carlo runner for a single temperature
def run_MC(T, steps, N=1000):
    n = np.ones((N, 3), int)
    eplot = []
    E = 3 * N * (np.pi ** 2) / 2

    for k in range(steps):
        i = randrange(N)
        j = randrange(3)

        if random() < 0.5:
            dn = 1
            dE = (2 * n[i, j] + 1) * (np.pi ** 2) / 2
        else:
            if n[i, j] == 1:
                dn = 0
                dE = 0
            else:
                dn = -1
                dE = -(2 * n[i, j] - 1) * (np.pi ** 2) / 2

        if dn != 0:
            if dE <= 0 or random() < np.exp(-dE / T):
                n[i, j] += dn
                E += dE

        eplot.append(E)

    return n, eplot


if __name__ == "__main__":
    os.makedirs("q1_plots", exist_ok=True)
    seed(1234)

    # --------------------------------------------
    # STEP 1 — T = 10 with 250k steps
    # --------------------------------------------
    T = 1600
    steps_T10 = 4000000

    print("Running T = 10 ...")
    n10, eplot10 = run_MC(T, steps=steps_T10)

    # Plot E vs time for T=10
    plt.figure(figsize=(8, 5))
    plt.plot(eplot10)
    plt.title(fr"Energy vs Monte Carlo Time ($k_B T = {T}$)")
    plt.xlabel("MC Step")
    plt.ylabel("Total Energy E")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"q1_plots/E_vs_time_T{T}.png", dpi=300)
    plt.clf()

    # Histogram for T = 10
    energy_n = n10[:, 0] ** 2 + n10[:, 1] ** 2 + n10[:, 2] ** 2
    unique_E, counts = np.unique(energy_n, return_counts=True)
    n_vals = np.sqrt(unique_E)

    plt.figure(figsize=(7, 5))
    plt.bar(n_vals, counts, width=0.1, edgecolor="black")
    plt.xlabel(r"$n = \sqrt{n_x^2 + n_y^2 + n_z^2}$")
    plt.ylabel("Frequency")
    plt.title(fr"Energy Level Distribution ($k_B T = {T}$)")
    plt.xticks(n_vals, [f"{val:.2f}" for val in n_vals], rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"q1_plots/histogram_T{T}.png", dpi=300)
    plt.clf()
    print(f"done for T={T} steps={steps_T10}")

    # --------------------------------------------
    # STEP 2 — RUN ALL TEMPERATURES WITH THEIR OWN STEP COUNTS
    # --------------------------------------------
    Ts = [10, 40, 100, 400, 1200, 1600]
    step_counts = {
        10: 500000,
        40: 500000,
        100: 1500000,
        400: 1500000,
        1200: 4000000,
        1600: 4000000
    }
    avg_after = {
        10: 200000,
        40: 200000,
        100: 800000,
        400: 800000,
        1200: 2500000,
        1600: 2500000
    }

    eplots = []
    E_eqs = []
    nT_vals = []

    for T in Ts:
        print(f"Running T = {T} ...")

        steps = step_counts[T]
        n, eplot = run_MC(T=T, steps=steps)
        eplots.append((T, eplot))

        # Compute equilibrium average
        discard = avg_after[T]
        E_eq = np.mean(eplot[discard:])
        E_eqs.append((T, E_eq))
        print(f"Done E for T = {T}")

        # ---- Compute average n using histogram formula ----
        energy_n = n[:, 0] ** 2 + n[:, 1] ** 2 + n[:, 2] ** 2
        unique_E, counts = np.unique(energy_n, return_counts=True)

        # Convert energy index to n-values
        n_vals = np.sqrt(unique_E)

        # Apply formula: n̄ = sum(counts * n_vals) / sum(counts)
        avg_n = np.sum(counts * n_vals) / np.sum(counts)

        nT_vals.append((T, avg_n))

        print(f"avg n for T = {T}  calculated")
    # --------------------------------------------
    # STEP 3 — Plot E(T) vs T
    # --------------------------------------------
    Ts_sorted = [x for x, y in E_eqs]
    E_sorted = [y for x, y in E_eqs]

    plt.figure()
    plt.plot(Ts_sorted, E_sorted, "o-", linewidth=2)
    plt.xlabel("Temperature T")
    plt.ylabel("Equilibrium Energy E(T)")
    plt.title("E(T) vs Temperature")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_plots/E_vs_T.png", dpi=300)
    plt.clf()

    print("\nE(T) values:")
    for T, Eeq in E_eqs:
        print(f"T={T}: E_eq = {Eeq:.3f}")
    print("-------Heat Capacity------------------------------------------------------------------")
    for i in range(len(E_eqs) - 1):
        del_T = (abs(E_eqs[i][0] - E_eqs[i + 1][0]))
        del_E = (abs(E_eqs[i][1] - E_eqs[i + 1][1]))
        print(f"For E_1 = {E_eqs[i][1]} E_2 = {E_eqs[i + 1][1]}\n T_1 = {E_eqs[i][0]} T_2 = {E_eqs[i + 1][0]}"
              f"\nDel_E = {del_E}; Del_T = {del_T} \n Heat Capacity = {del_E / del_T} \n\n\n")
    print("----------------------------------------------------------------------------------------")

    # --------------------------------------------
    # Plot n(T) vs T
    # --------------------------------------------
    Ts_n = [x for x, y in nT_vals]
    avg_ns = [y for x, y in nT_vals]

    plt.figure(figsize=(7, 5))
    plt.plot(Ts_n, avg_ns, "o-", linewidth=2)
    plt.xlabel("Temperature T")
    plt.ylabel(r"Average $\,\bar{n}(T)$")
    plt.title(r"$\bar{n}(T)$ vs Temperature")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_plots/n_vs_T.png", dpi=300)
    plt.clf()

    print("\nn(T) values:")
    for T, navg in nT_vals:
        print(f"T={T}: n_avg = {navg:.3f}")
