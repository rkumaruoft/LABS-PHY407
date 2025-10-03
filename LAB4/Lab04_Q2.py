from scipy import constants
import numpy as np

# constants
a = 10 * constants.electron_volt  # J
L = 5e-10  # meters
h_bar = constants.hbar  # Joule sec
m_e = constants.electron_mass  # kg
q_e = constants.elementary_charge  # coulomb


def H_matrix_element(m, n):
    # if m != n and both even or odd
    if m != n:
        if m % 2 == n % 2:
            return 0
        else:
            numerator = -8 * a * m * n
            denominator = (np.pi * ((m ** 2) - (n ** 2))) ** 2
            return numerator / denominator
    else:
        term_1 = a / 2
        term_2 = (np.pi ** 2) * (h_bar ** 2) * (m ** 2) / (2 * m_e * (L ** 2))
        return term_1 + term_2


if __name__ == "__main__":

    # Part c
    m_max = 10
    n_max = 10

    H_matrix = np.zeros((m_max, n_max))
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            H_matrix[m - 1, n - 1] = H_matrix_element(m, n)

    eigenvals, eigenvecs = np.linalg.eigh(H_matrix)

    print("------------------------------Part C -------------------------")
    for i in range(0, 10):
        this_energy = eigenvals[i] / constants.electron_volt
        print(f"Energy E{i} = {this_energy} eV")

    # Part d
    m_max = 100
    n_max = 100

    H_matrix = np.zeros((m_max, n_max))
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            H_matrix[m - 1, n - 1] = H_matrix_element(m, n)

    eigenvals, eigenvecs = np.linalg.eigh(H_matrix)

    print("------------------------------Part D -------------------------")
    for i in range(0, 10):
        this_energy = eigenvals[i] / constants.electron_volt
        print(f"Energy E{i} = {this_energy} eV")
