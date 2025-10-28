import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

# constants
hbar = 1.054571817e-34
eV = 1.602176634e-19
m_e = 9.10938356e-31

# parameters
V0_eV = 50.0
V0 = V0_eV * eV
a = 1e-11

# grid
N = 400000
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


psi0 = eigvecs[:, 0].copy()
psi1 = eigvecs[:, 1].copy()
psi2 = eigvecs[:, 2].copy()


ix = slice(1, N)
x_interior = x[ix]
dx_interior = x_interior[1] - x_interior[0]

# symmetry-based normalization
mid_idx = len(x_interior) // 2
left_x = x_interior[:mid_idx+1]
# functions to compute normalization
def normalize_by_symmetry(psi_full):
    psi = psi_full[ix].copy()
    psi = np.real(psi)
    # left half corresponding to left_x
    psi_left = psi[:mid_idx+1]
    integral_left = np.trapz(psi_left**2, left_x)
    integral_total = 2.0 * integral_left
    norm = np.sqrt(integral_total)
    return psi / norm

psi0 = normalize_by_symmetry(psi0)
psi1 = normalize_by_symmetry(psi1)
psi2 = normalize_by_symmetry(psi2)

center_idx = (N - 1) // 2
if psi0[center_idx] < 0: psi0 *= -1
if psi1[center_idx + 5] < 0: psi1 *= -1
if psi2[center_idx] < 0: psi2 *= -1

# plotting range near origin
spread = 7.0
plot_x_min = -spread * a
plot_x_max = +spread * a
i_min = np.searchsorted(x, plot_x_min)
i_max = np.searchsorted(x, plot_x_max)

# plot normalized wavefunctions on same axes
plt.figure(figsize=(8,5))
plt.plot(x[i_min:i_max], psi0[i_min:i_max], label=f"n=0, E={E0:.3f} eV")
plt.plot(x[i_min:i_max], psi1[i_min:i_max], label=f"n=1, E={E1:.3f} eV")
plt.plot(x[i_min:i_max], psi2[i_min:i_max], label=f"n=2, E={E2:.3f} eV")
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("x (m)")
plt.ylabel(r"$|\psi_n(x)|$")
print("Normalized wavefunctions for quartic anharmonic oscillator")
plt.legend()
plt.tight_layout()
plt.show()
