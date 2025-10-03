from scipy import constants
import numpy as np

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

    H_matrix = np.zeros((m_max, n_max))
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            H_matrix[m - 1, n - 1] = H_matrix_element(m, n)

    eigenvals, eigenvecs = np.linalg.eigh(H_matrix)

    for i in range(0, 10):
        print()
