import numpy as np
import matplotlib.pyplot as plt

kOverM = 400.0
dt = 0.001
x0_init = 0.10
v0_init = 0.0

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
    vals, vecs = np.linalg.eigh(L)  # symmetric
    omegas = np.sqrt(kOverM * vals)
    freqs_hz = omegas / (2 * np.pi)
    return vals, vecs, omegas, freqs_hz

def acceleration_from_positions(x):
    x = np.asarray(x, dtype=float).ravel()
    xp = np.concatenate((np.array([0.0]), x, np.array([0.0])))
    a = -kOverM * (2.0 * xp[1:-1] - xp[0:-2] - xp[2:])
    return a

def velocity_verlet(N, T_total):
    Nt = int(np.ceil(T_total / dt))
    t = np.linspace(0.0, Nt * dt, Nt + 1)
    x = np.zeros((Nt + 1, N))
    v = np.zeros((Nt + 1, N))

    x[0, 0] = x0_init
    v[0, :] = v0_init
    a = acceleration_from_positions(x[0])

    for n in range(Nt):
        x[n + 1] = x[n] + v[n] * dt + 0.5 * a * dt * dt
        a_new = acceleration_from_positions(x[n + 1])
        v[n + 1] = v[n] + 0.5 * (a + a_new) * dt
        a = a_new

    return t, x, v

def plot_time_series(t, x, title, time_window=None, figsize=(10, 6)):
    N = x.shape[1]
    if time_window is not None:
        tmin, tmax = time_window
        idx = np.where((t >= tmin) & (t <= tmax))[0]
        tplt = t[idx]
        xplt = x[idx]
    else:
        tplt = t
        xplt = x

    plt.figure(figsize=figsize)
    for i in range(N):
        plt.plot(tplt, xplt[:, i], label=f'floor {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()

def run_for_N(N):
    print(f'\n------N = {N}-------')
    vals, modes, omegas, freqs_hz = eigen_analysis(N)

    print('The first five eigenfrequencies (Hz):', np.around(freqs_hz[:5], 3))

    positive_omegas = omegas[omegas > 1e-12]
    if positive_omegas.size == 0:
        omega_min = omegas[0]
    else:
        omega_min = positive_omegas.min()
    period_min = 2 * np.pi / omega_min
    short_T = max(5 * period_min, 0.5)
    long_T = max(50 * period_min, 5.0)
    print(f'Using short T = {short_T:.3f} s and long T = {long_T:.3f} s for plots.')

    t_short, x_short, _ = velocity_verlet(N, short_T)
    t_long, x_long, _ = velocity_verlet(N, long_T)

    plot_time_series(t_short, x_short, f'N={N} All floors, short time ({short_T:.3f} s)',
                     time_window=(t_short[0], t_short[-1]))
    plot_time_series(t_long, x_long, f'N={N} All floors, long time ({long_T:.3f} s)',
                     time_window=(t_long[0], t_long[-1]))

    init_x = np.zeros(N)
    init_x[0] = x0_init

    coeffs = modes.T @ init_x
    amps = np.abs(coeffs)

    plt.figure(figsize=(6, 4))
    markerline, stemlines, baseline = plt.stem(np.arange(1, N + 1), amps, basefmt=" ")
    plt.setp(markerline, marker='o', markersize=6, markerfacecolor='C0')
    plt.setp(stemlines, linestyle='-', color='C0')
    plt.setp(baseline, visible=False)
    plt.xlabel('Mode index')
    plt.ylabel('Modal amplitude (abs)')
    plt.title(f'N={N} Modal amplitudes from initial displacement of bottom floor')
    plt.tight_layout()
    plt.show()

    return (vals, modes, omegas, freqs_hz), (t_short, x_short), (t_long, x_long)

if __name__ == '__main__':
    Ns = [3, 10]
    results = {}
    for N in Ns:
        res = run_for_N(N)
        results[N] = res
    print('\nFinished.')
