import numpy as np
from scipy.linalg import eigh_tridiagonal

# constants
hbar = 1.054571817e-34
eV = 1.602176634e-19
m_e = 9.10938356e-31

# parameters
V0_eV = 50.0
V0 = V0_eV * eV
a = 1e-11

# grid
N = 200000
x_min = -10.0 * a
x_max = +10.0 * a
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# potential (quartic)
V = V0 * (x**4) / (a**4)

# finite-difference second derivative
coeff = hbar**2 / (2.0 * m_e * dx**2)
diag = (2.0 * coeff) + V
off = -coeff * np.ones(N-1)

# solve for lowest eigenvalues
num_eigs = 6
eigvals, eigvecs = eigh_tridiagonal(diag, off, select='i', select_range=(0, num_eigs-1))

# convert eigenvalues to eV
eig_eV = eigvals / eV

E0, E1, E2 = eig_eV[0], eig_eV[1], eig_eV[2]
print("E0 = {:.6f} eV".format(E0))
print("E1 = {:.6f} eV".format(E1))
print("E2 = {:.6f} eV".format(E2))
print("Spacing E1-E0 = {:.6f} eV".format(E1-E0))
print("Spacing E2-E1 = {:.6f} eV".format(E2-E1))
