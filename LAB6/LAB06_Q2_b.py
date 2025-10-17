import numpy as np
import matplotlib.pyplot as plt

#Parameters
kOverM = 400.0
dt = 0.001
mode_amp = 0.10

def stiffness_matrix(N):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        if i > 0:
            L[i, i - 1] = -1.0
        if i < N - 1:
            L[i, i + 1] = -1.0
    return L

def eigen_analysis(N):
    L = stiffness_matrix(N)
    vals, vecs = np.linalg.eigh(L)
    omegas = np.sqrt(kOverM * vals)
    freqs_hz = omegas / (2 * np.pi)
    return vals, vecs, omegas, freqs_hz

def acceleration_from_positions(x):
    x = np.asarray(x, dtype=float).ravel()
    xp = np.concatenate((np.array([0.0]), x, np.array([0.0])))
    a = -kOverM * (2.0 * xp[1:-1] - xp[0:-2] - xp[2:])
    return a

def velocity_verlet(N, T_total, x0_vec, v0_vec=None):
    Nt = int(np.ceil(T_total / dt))
    t = np.linspace(0.0, Nt * dt, Nt + 1)
    x = np.zeros((Nt + 1, N))
    v = np.zeros((Nt + 1, N))

    x[0] = x0_vec.copy()
    if v0_vec is None:
        v[0] = np.zeros(N)
    else:
        v[0] = v0_vec.copy()

    a = acceleration_from_positions(x[0])
    for n in range(Nt):
        x[n + 1] = x[n] + v[n] * dt + 0.5 * a * dt * dt
        a_new = acceleration_from_positions(x[n + 1])
        v[n + 1] = v[n] + 0.5 * (a + a_new) * dt
        a = a_new
    return t, x, v

def plot_time_series(t, x, title, figsize=(8, 5)):
    N = x.shape[1]
    plt.figure(figsize=figsize)
    for i in range(N):
        plt.plot(t, x[:, i], label=f'floor {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    print("Caption: ", title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    N = 3
    vals, modes, omegas, freqs_hz = eigen_analysis(N)

    print("Eigenvalues (lambda):", np.round(vals, 6))
    print("Angular freqs (rad/s):", np.round(omegas, 6))
    print("Frequencies (Hz):", np.round(freqs_hz, 6))

    #slowest nonzero mode
    positive_omegas = omegas[omegas > 1e-12]
    omega_min = positive_omegas.min()
    period_min = 2 * np.pi / omega_min

    #simulate few cycles
    T_sim = max(8 * period_min, 1.0)

    for mode_idx in range(N):
        eigvec = modes[:, mode_idx]

        #Normalize eigenvector sign and scale
        eigvec = eigvec / np.max(np.abs(eigvec)) * mode_amp

        x0 = eigvec.copy()
        v0 = np.zeros(N)

        t, x, v = velocity_verlet(N, T_sim, x0, v0)

        #Plot whole time series
        plot_time_series(t, x, f'N={N} Mode {mode_idx + 1} (init from eigenvector), f = {freqs_hz[mode_idx]:.6f} Hz')

        #projection vs time to show single-mode oscillation
        phi_j = modes[:, mode_idx]
        q = x @ phi_j
        plt.figure(figsize=(6, 3))
        plt.plot(t, q)
        plt.xlabel('Time (s)')
        plt.ylabel('Modal coordinate (projection)')
        print("Caption: ", f'Modal projection onto eigenvector {mode_idx + 1}')
        plt.tight_layout()
        plt.show()
