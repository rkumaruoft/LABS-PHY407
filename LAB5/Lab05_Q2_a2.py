import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from common_funcs import gaussian_quad

def period_integrand(x0, m_param, k_param):
    V0 = 0.5 * k_param * x0**2
    mc2 = m_param * c**2

    def integrand(x):
        xa = np.asarray(x, dtype=float)
        Vx = 0.5 * k_param * xa**2
        E_minus_V = mc2 + V0 - Vx
        gamma = np.maximum(E_minus_V / mc2, 1.0)
        inside = 1.0 - 1.0 / (gamma * gamma)
        inside = np.clip(inside, 0.0, 1.0)
        v = c * np.sqrt(inside)
        with np.errstate(divide='ignore'):
            inv = 1.0 / v
        if np.isscalar(x):
            return float(inv) if np.isfinite(inv) else 1e300
        inv = np.array(inv, dtype=float)
        inv[np.isinf(inv)] = 1e300
        return inv

    return integrand

def v_from_p(p, m_param):
    return p / (m_param * np.sqrt(1.0 + (p / (m_param * c))**2))

def euler_cromer_relativistic(x0, dt, nt, m_param, k_param):
    x = np.zeros(nt, dtype=float)
    p = np.zeros(nt, dtype=float)
    v = np.zeros(nt, dtype=float)
    x[0] = x0
    for n in range(nt - 1):
        F = -k_param * x[n]
        p[n + 1] = p[n] + F * dt
        v[n + 1] = v_from_p(p[n + 1], m_param)
        x[n + 1] = x[n] + v[n + 1] * dt
    return x, v, p

def compute_fft_norm(signal, dt, p_factor=4):
    s = signal - np.mean(signal)
    nfft = int(2 ** np.ceil(np.log2(len(s) * p_factor)))
    S = np.fft.rfft(s, n=nfft)
    freqs = np.fft.rfftfreq(nfft, dt)
    amp = np.abs(S)
    amp_norm = amp / amp.max() if amp.max() != 0 else amp

    return freqs, amp, amp_norm

if __name__ == "__main__":
    m = 1.0
    k = 1.0
    omega = np.sqrt(k / m)
    T_lin = 2 * (pi / omega)
    xc = c / omega

    periods_to_record = 30
    dt = T_lin / 2000.0
    t_total = periods_to_record * T_lin
    nt = int(np.ceil(t_total / dt))
    time = np.arange(nt) * dt

    x0_1 = 1.0
    x0_2 = xc
    x0_3 = 10.0 * xc

    x1, v1, p1 = euler_cromer_relativistic(x0_1, dt, nt, m, k)
    x2, v2, p2 = euler_cromer_relativistic(x0_2, dt, nt, m, k)
    x3, v3, p3 = euler_cromer_relativistic(x0_3, dt, nt, m, k)

    freqs_x1, amp_x1, amp_x1_norm = compute_fft_norm(x1, dt)
    freqs_x2, amp_x2, amp_x2_norm = compute_fft_norm(x2, dt)
    freqs_x3, amp_x3, amp_x3_norm = compute_fft_norm(x3, dt)
    freqs_v1, amp_v1, amp_v1_norm = compute_fft_norm(v1, dt)
    freqs_v2, amp_v2, amp_v2_norm = compute_fft_norm(v2, dt)
    freqs_v3, amp_v3, amp_v3_norm = compute_fft_norm(v3, dt)

    N_quad = 300
    def compute_T_by_gauss(x0, m_param=m, k_param=k, nquad=N_quad):
        if x0 <= 0:
            return np.nan
        integrand = period_integrand(x0, m_param, k_param)
        return 4.0 * gaussian_quad(integrand, nquad, 0.0, x0)

    T_est_1 = compute_T_by_gauss(x0_1)
    T_est_2 = compute_T_by_gauss(x0_2)
    T_est_3 = compute_T_by_gauss(x0_3)

    f_T1 = 1.0 / T_est_1 if (T_est_1 and T_est_1 > 0) else np.nan
    f_T2 = 1.0 / T_est_2 if (T_est_2 and T_est_2 > 0) else np.nan
    f_T3 = 1.0 / T_est_3 if (T_est_3 and T_est_3 > 0) else np.nan

    print("Gaussian-quadrature period estimates:")
    print(f" x0 = {x0_1:.3e} -> T = {T_est_1:.6e} s, 1/T = {f_T1:.6e} Hz")
    print(f" x0 = {x0_2:.3e} -> T = {T_est_2:.6e} s, 1/T = {f_T2:.6e} Hz")
    print(f" x0 = {x0_3:.3e} -> T = {T_est_3:.6e} s, 1/T = {f_T3:.6e} Hz")
    print(f"Linear small-amplitude T_lin = {T_lin:.6e} s, 1/T_lin = {1.0/T_lin:.6e} Hz")

    plt.figure()
    plt.plot(freqs_x1, amp_x1_norm, label=f'x(t), x0={x0_1}', linewidth=1.0)
    plt.plot(freqs_x2, amp_x2_norm, label='x(t), x0=xc', linewidth=1.0)
    plt.plot(freqs_x3, amp_x3_norm, label='x(t), x0=10xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * np.pi))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized amplitude |x̂(f)| / |x̂|max')
    if not np.isnan(f_T1):
        plt.axvline(f_T1, color='C0', linestyle='--', linewidth=1.2, label=f'1/T_quad (x0={x0_1}) = {f_T1:.4e} Hz')
    if not np.isnan(f_T2):
        plt.axvline(f_T2, color='C1', linestyle='--', linewidth=1.2, label=f'1/T_quad (x0=xc) = {f_T2:.4e} Hz')
    if not np.isnan(f_T3):
        plt.axvline(f_T3, color='C2', linestyle='--', linewidth=1.2, label=f'1/T_quad (x0=10xc) = {f_T3:.4e} Hz')
    pk1 = freqs_x1[np.argmax(amp_x1)]
    pk2 = freqs_x2[np.argmax(amp_x2)]
    pk3 = freqs_x3[np.argmax(amp_x3)]
    plt.scatter([pk1, pk2, pk3], [amp_x1_norm.max(), amp_x2_norm.max(), amp_x3_norm.max()],
                color=['C0', 'C1', 'C2'], marker='x', zorder=10, label='FFT peak freq')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(freqs_v1, amp_v1_norm, label=f'v(t), x0={x0_1}', linewidth=1.0)
    plt.plot(freqs_v2, amp_v2_norm, label='v(t), x0=xc', linewidth=1.0)
    plt.plot(freqs_v3, amp_v3_norm, label='v(t), x0=10xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * np.pi))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized amplitude |v̂(f)| / |v̂|max')
    if not np.isnan(f_T1):
        plt.axvline(f_T1, color='C0', linestyle='--', linewidth=1.2)
    if not np.isnan(f_T2):
        plt.axvline(f_T2, color='C1', linestyle='--', linewidth=1.2)
    if not np.isnan(f_T3):
        plt.axvline(f_T3, color='C2', linestyle='--', linewidth=1.2)
    pk1_v = freqs_v1[np.argmax(amp_v1)]
    pk2_v = freqs_v2[np.argmax(amp_v2)]
    pk3_v = freqs_v3[np.argmax(amp_v3)]
    plt.scatter([pk1_v, pk2_v, pk3_v], [amp_v1_norm.max(), amp_v2_norm.max(), amp_v3_norm.max()],
                color=['C0', 'C1', 'C2'], marker='x', zorder=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
