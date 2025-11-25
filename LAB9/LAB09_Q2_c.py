import numpy as np
import scipy.constants as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Parameters
L = 1e-8                       # m
m = sc.m_e                     # electron mass
P = 1024                       # number of spatial segments
Nx = P + 1                     # number of grid points
x = np.linspace(-L/2, L/2, Nx)
dx = x[1] - x[0]

tau = 1e-18                    # time step (s)
Nsteps = 4000                  # number of time steps
T = Nsteps * tau

# Wavepacket parameters
sigma = L / 25.0
kappa = 500.0 / L
x0 = L / 5.0

# Potential: harmonic oscillator V = 1/2 m ω^2 x^2
omega = 3.0e15                 # rad/s
V = 0.5 * m * omega**2 * x**2

# Create discrete Laplacian with Dirichlet BCs (interior points only)
hbar = sc.hbar

# Second derivative operator (finite differences) with Dirichlet BCs
diag = np.zeros(Nx)
off = np.zeros(Nx-1)
diag[:] = -2.0
off[:] = 1.0

Lapl = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csc') / dx**2

# Hamiltonian H = - (hbar^2)/(2m) Laplacian + V (diagonal)
H = (-(hbar**2) / (2.0 * m)) * Lapl + sp.diags(V, 0, format='csc')

# Crank-Nicolson matrices:
I = sp.eye(Nx, format='csc')
coeff = 1j * tau / (2.0 * hbar)
A = (I + coeff * H)
B = (I - coeff * H)

# Modify A and B so boundary rows enforce psi[0]=0 and psi[-1]=0
def enforce_dirichlet_on_matrix(M):
    M = M.tolil()
    M[0, :] = 0
    M[-1, :] = 0
    M[0, 0] = 1.0
    M[-1, -1] = 1.0
    return M.tocsc()

A = enforce_dirichlet_on_matrix(A)
B = enforce_dirichlet_on_matrix(B)

# Pre-factorize A for speed
A_factor = spla.factorized(A)

# Initial psi (unnormalized)
psi0 = np.exp(- (x - x0)**2 / (4.0 * sigma**2) + 1j * kappa * x)
# enforce psi=0 at boundaries
psi0[0] = 0.0
psi0[-1] = 0.0

# Normalize psi0 numerically using trapezoidal rule
def norm(psi):
    return np.sqrt(np.trapezoid(np.abs(psi)**2, x))

psi0 = psi0 / norm(psi0)

# diagnostics functions
def expectation_x(psi):
    return np.trapezoid(np.conjugate(psi) * x * psi, x).real

# compute expectation value of energy E = <psi|H|psi>
def expectation_energy(psi):
    Hpsi = H.dot(psi)
    return np.trapezoid(np.conjugate(psi) * Hpsi, x).real

# time-stepping
psi = psi0.copy()
times = [0.0]
norms = [np.trapezoid(np.abs(psi)**2, x)]
Evals = [expectation_energy(psi)]
xexp = [expectation_x(psi)]

# choose times at which to store/plot
store_every = int(Nsteps/4)
store_indices = list(range(0, Nsteps+1, store_every))
stored_psis = []
stored_times = []

print("Starting time evolution: Nsteps =", Nsteps)
for n in range(1, Nsteps+1):
    b = B.dot(psi)
    # enforce boundary values on rhs to keep them zero
    b[0] = 0.0
    b[-1] = 0.0
    # solve A psi_new = b
    psi = A_factor(b)
    # enforce exact zero at boundaries
    psi[0] = 0.0
    psi[-1] = 0.0

    if n in store_indices:
        t = n * tau
        times.append(t)
        norms.append(np.trapezoid(np.abs(psi)**2, x))
        Evals.append(expectation_energy(psi))
        xexp.append(expectation_x(psi))
        stored_psis.append(psi.copy())
        stored_times.append(t)
        print(f"step {n:5d}, t = {t:.3e} s, norm = {norms[-1]:.8f}, <x> = {xexp[-1]:.3e} m, E = {Evals[-1]:.3e} J")

print("Evolution complete.")

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(times, norms, '-o', markersize=4)
plt.xlabel('time [s]')
plt.ylabel('Normalization')
plt.title('Normalization vs time')

plt.subplot(1, 2, 2)
plt.plot(times, Evals, '-o', markersize=4)
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.title('Energy vs time')
plt.tight_layout()
plt.show()

# Plot probability density at selected times
plt.figure(figsize=(10, 6))
for i, psi_s in enumerate(stored_psis):
    plt.plot(x * 1e9, np.abs(psi_s)**2, label=f"t={stored_times[i]*1e15:.1f} fs")
plt.xlabel('x [nm]')
plt.ylabel(r'$|\psi|^2$')
plt.legend()
plt.title('Probability density at selected times')
plt.show()
