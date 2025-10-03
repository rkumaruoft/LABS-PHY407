from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

from Lab04_Q2 import H_matrix_element

# constants
a = 10 * constants.electron_volt  # J
L = 5e-10  # meters
h_bar = constants.hbar  # Joule sec
m_e = constants.electron_mass  # kg
q_e = constants.elementary_charge  # coulomb

if __name__ == "__main__":
    m_max = 100
    n_max = 100

    # build the H matrix
    H_matrix = np.zeros((m_max, n_max))
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            H_matrix[m - 1, n - 1] = H_matrix_element(m, n)

    # compute the eigenvectors and eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(H_matrix)

    # ground state eigenvectors
    ground_state_coeffs = eigenvecs[:, 0]

    x = np.linspace(0, L, 4000)

    # construct the ground state
    psi_0 = np.zeros_like(x)
    for i in range(len(ground_state_coeffs)):
        psi_0 += ground_state_coeffs[i] * np.sin(np.pi * (i + 1) * x / L)

    # check normalization
    print("---------Ground State---------------------------------------------")
    norm = np.trapezoid(np.abs(psi_0) ** 2, x)
    print("Normalization check (before):", norm)
    psi_0 /= np.sqrt(norm)
    print("Normalization check (after):", np.trapezoid(np.abs(psi_0) ** 2, x))

    # first exited state
    first_state_coeffs = eigenvecs[:, 1]
    # construct the first state
    psi_1 = np.zeros_like(x)
    for i in range(len(first_state_coeffs)):
        psi_1 += first_state_coeffs[i] * np.sin(np.pi * (i + 1) * x / L)

    # check normalization
    print("---------First Exited State---------------------------------------------")
    norm = np.trapezoid(np.abs(psi_1) ** 2, x)
    print("Normalization check (before):", norm)
    psi_1 /= np.sqrt(norm)
    print("Normalization check (after):", np.trapezoid(np.abs(psi_1) ** 2, x))

    # Second exited state
    second_state_coeffs = eigenvecs[:, 2]
    # construct the second state
    psi_2 = np.zeros_like(x)
    for i in range(len(second_state_coeffs)):
        psi_2 += second_state_coeffs[i] * np.sin(np.pi * (i + 1) * x / L)

    # check normalization
    print("---------First Exited State---------------------------------------------")
    norm = np.trapezoid(np.abs(psi_2) ** 2, x)
    print("Normalization check (before):", norm)
    psi_2 /= np.sqrt(norm)
    print("Normalization check (after):", np.trapezoid(np.abs(psi_1) ** 2, x))

    psi_0_sq = np.abs(psi_0) ** 2
    psi_1_sq = np.abs(psi_1) ** 2
    psi_2_sq = np.abs(psi_2) ** 2
    plt.plot(x, psi_0_sq, color="blue", label=r"$|\psi_0(x)|^2$")
    plt.plot(x, psi_1_sq, color="red", label=r"$|\psi_1(x)|^2$")
    plt.plot(x, psi_2_sq, color="green", label=r"$|\psi_2(x)|^2$")
    plt.xlabel("x (m)")
    plt.ylabel(r"Probability Density $|\psi_n(x)|^2$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Probability Density Plot", dpi=300)
    plt.show()
