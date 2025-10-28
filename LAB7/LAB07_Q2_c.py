import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy import special
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

#------------------------------------------------------------------------------
# QUARTIC (anharmonic) potential and solution

# potential (quartic)
V_q = V0 * (x**4) / (a**4)

# finite-difference second derivative (quartic)
coeff = hbar**2 / (2.0 * m_e * dx**2)
diag_q = (2.0 * coeff) + V_q
off_q = -coeff * np.ones(N-1)

# solve for lowest eigenvalues (quartic)
num_eigs = 6
eigvals_q, eigvecs_q = eigh_tridiagonal(diag_q, off_q, select='i', select_range=(0, num_eigs-1))

# convert eigenvalues to eV (quartic)
eig_eV_q = eigvals_q / eV
E0_q, E1_q, E2_q = eig_eV_q[0], eig_eV_q[1], eig_eV_q[2]
print("Quartic: E0 = {:.6f} eV".format(E0_q))
print("Quartic: E1 = {:.6f} eV".format(E1_q))
print("Quartic: E2 = {:.6f} eV".format(E2_q))
print("Quartic spacing E1-E0 = {:.6f} eV".format(E1_q-E0_q))
print("Quartic spacing E2-E1 = {:.6f} eV".format(E2_q-E1_q))

# eigenvectors (quartic)
psi_q0 = eigvecs_q[:, 0].copy()
psi_q1 = eigvecs_q[:, 1].copy()
psi_q2 = eigvecs_q[:, 2].copy()

ix = slice(1, N)
x_interior = x[ix]
dx_interior = x_interior[1] - x_interior[0]

mid_idx = len(x_interior) // 2
left_x = x_interior[:mid_idx+1]

def normalize_by_symmetry(psi_full):
    psi = psi_full[ix].copy()
    psi = np.real(psi)
    psi_left = psi[:mid_idx+1]
    # use np.trapezoid to avoid DeprecationWarning
    integral_left = np.trapezoid(psi_left**2, left_x)
    integral_total = 2.0 * integral_left
    norm = np.sqrt(integral_total)
    return psi / norm

psi_q0 = normalize_by_symmetry(psi_q0)
psi_q1 = normalize_by_symmetry(psi_q1)
psi_q2 = normalize_by_symmetry(psi_q2)

center_idx = (N - 1) // 2
if psi_q0[center_idx] < 0: psi_q0 *= -1
if psi_q1[center_idx + 5] < 0: psi_q1 *= -1
if psi_q2[center_idx] < 0: psi_q2 *= -1

#------------------------------------------------------------------------------
# HARMONIC potential and solution

# potential (harmonic)
V_h = V0 * (x**2) / (a**2)

# finite-difference second derivative (harmonic)
diag_h = (2.0 * coeff) + V_h
off_h = -coeff * np.ones(N-1)

# solve for lowest eigenvalues (harmonic)
eigvals_h, eigvecs_h = eigh_tridiagonal(diag_h, off_h, select='i', select_range=(0, num_eigs-1))

# convert eigenvalues to eV (harmonic)
eig_eV_h = eigvals_h / eV
E0_h, E1_h, E2_h = eig_eV_h[0], eig_eV_h[1], eig_eV_h[2]
print("Harmonic: E0 = {:.6f} eV".format(E0_h))
print("Harmonic: E1 = {:.6f} eV".format(E1_h))
print("Harmonic: E2 = {:.6f} eV".format(E2_h))
print("Harmonic spacing E1-E0 = {:.6f} eV".format(E1_h-E0_h))
print("Harmonic spacing E2-E1 = {:.6f} eV".format(E2_h-E1_h))

# eigenvectors (harmonic)
psi_h0 = eigvecs_h[:, 0].copy()
psi_h1 = eigvecs_h[:, 1].copy()
psi_h2 = eigvecs_h[:, 2].copy()

# normalize harmonic eigenvectors
psi_h0 = normalize_by_symmetry(psi_h0)
psi_h1 = normalize_by_symmetry(psi_h1)
psi_h2 = normalize_by_symmetry(psi_h2)

if psi_h0[center_idx] < 0: psi_h0 *= -1
if psi_h1[center_idx + 5] < 0: psi_h1 *= -1
if psi_h2[center_idx] < 0: psi_h2 *= -1

#------------------------------------------------------------------------------
# Analytical harmonic oscillator wavefunctions for n=0,1,2 (to overlay)

# compute omega from matching V(x) = 1/2 m omega^2 x^2 = V0 x^2 / a^2
omega = np.sqrt(2.0 * V0 / (m_e * a**2))

def psi_analytic(n, x_phys):
    """Analytical harmonic oscillator wavefunction psi_n(x) (real) for n=0,1,2."""
    xi_local = np.sqrt(m_e * omega / hbar) * x_phys
    Hn = special.eval_hermite(n, xi_local)
    prefac = (m_e * omega / (np.pi * hbar))**0.25
    norm_n = 1.0 / np.sqrt((2.0**n) * special.factorial(n))
    psi = prefac * norm_n * Hn * np.exp(-0.5 * xi_local**2)
    return psi

# compute analytic wavefunctions on interior grid
psi_a0 = psi_analytic(0, x_interior)
psi_a1 = psi_analytic(1, x_interior)
psi_a2 = psi_analytic(2, x_interior)

# analytical energies for reference (eV)
E_a0 = 0.5 * hbar * omega / eV
E_a1 = 1.5 * hbar * omega / eV
E_a2 = 2.5 * hbar * omega / eV
print("Analytical harmonic energies (eV): E0={:.6f}, E1={:.6f}, E2={:.6f}".format(E_a0, E_a1, E_a2))

# align signs for plotting (make central value positive for even states)
center_in = len(x_interior) // 2
if psi_a0[center_in] < 0: psi_a0 *= -1
if psi_a2[center_in] < 0: psi_a2 *= -1
if psi_a1[center_in + 5] < 0: psi_a1 *= -1

#------------------------------------------------------------------------------
# Plotting:
spread = 7.0
plot_x_min = -spread * a
plot_x_max = +spread * a
i_min = np.searchsorted(x_interior, plot_x_min)
i_max = np.searchsorted(x_interior, plot_x_max)

# harmonic plot (numeric + analytical overlays)
plt.figure()
plt.plot(x_interior[i_min:i_max], psi_h0[i_min:i_max], label=f"n=0 numeric, E={E0_h:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_h1[i_min:i_max], label=f"n=1 numeric, E={E1_h:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_h2[i_min:i_max], label=f"n=2 numeric, E={E2_h:.3f} eV")
# overlay analytical (dashed)
plt.plot(x_interior[i_min:i_max], psi_a0[i_min:i_max], 'k--', label=f"n=0 analytic, E={E_a0:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_a1[i_min:i_max], 'C1--', label=f"n=1 analytic, E={E_a1:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_a2[i_min:i_max], 'C2--', label=f"n=2 analytic, E={E_a2:.3f} eV")
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("x (m)")
plt.ylabel(r"$\psi_n(x)$")
print("Harmonic oscillator: numeric vs analytic")
plt.legend()
plt.tight_layout()
plt.show()

# quartic plot (numeric only)
plt.figure()
plt.plot(x_interior[i_min:i_max], psi_q0[i_min:i_max], label=f"n=0, E={E0_q:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_q1[i_min:i_max], label=f"n=1, E={E1_q:.3f} eV")
plt.plot(x_interior[i_min:i_max], psi_q2[i_min:i_max], label=f"n=2, E={E2_q:.3f} eV")
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("x (m)")
plt.ylabel(r"$\psi_n(x)$")
print("Quartic anharmonic oscillator")
plt.legend()
plt.tight_layout()
plt.show()
