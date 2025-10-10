"""Re-revisiting the relativistic spring"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *

def v_from_p(p, m):
    return p / (m * np.sqrt(1.0 + (p / (m * c))**2))

def euler_cromer_relativistic(x0, dt, nt, m, k):
    x = np.zeros(nt, dtype=float)
    p = np.zeros(nt, dtype=float)
    v = np.zeros(nt, dtype=float)
    x[0] = x0
    p[0] = 0.0
    v[0] = 0.0
    for n in range(nt - 1):
        F = -k * x[n]
        p[n + 1] = p[n] + F * dt
        v[n + 1] = v_from_p(p[n + 1], m)
        x[n + 1] = x[n] + v[n + 1] * dt
    return x, v, p

if __name__ == "__main__":
    #Parameters:
    m = 1.0
    k = 1.0

    omega = np.sqrt(k / m)
    T = 2 * (pi / omega)
    xc = c / omega

    periods_to_record = 30
    dt = T / 2000.0
    t_total = periods_to_record * T
    nt = int(np.ceil(t_total / dt))
    time = np.arange(nt) * dt

    section = 3 * T
    n_short = int(min(nt, max(10, int(section / dt))))
    short_slice = slice(0, n_short)

    #Initial conditions
    x0_1 = 1.0
    x0_2 = xc
    x0_3 = 10.0 * xc

    x1, v1, p1 = euler_cromer_relativistic(x0_1, dt, nt, m, k)
    x2, v2, p2 = euler_cromer_relativistic(x0_2, dt, nt, m, k)
    x3, v3, p3 = euler_cromer_relativistic(x0_3, dt, nt, m, k)

    #FFT
    p_factor = 4
    def compute_fft_norm(signal, dt, p_factor=4):
        s = signal - np.mean(signal)
        nfft = int(2 ** np.ceil(np.log2(len(s) * p_factor)))
        S = np.fft.rfft(s, n=nfft)
        freqs = np.fft.rfftfreq(nfft, dt)
        amp = np.abs(S)
        amp_norm = amp / amp.max() if amp.max() != 0 else amp
        return freqs, amp, amp_norm

    #FFTs for positions
    freqs_x1, amp_x1, amp_x1_norm = compute_fft_norm(x1, dt, p_factor)
    freqs_x2, amp_x2, amp_x2_norm = compute_fft_norm(x2, dt, p_factor)
    freqs_x3, amp_x3, amp_x3_norm = compute_fft_norm(x3, dt, p_factor)

    #FFTs for velocities
    freqs_v1, amp_v1, amp_v1_norm = compute_fft_norm(v1, dt, p_factor)
    freqs_v2, amp_v2, amp_v2_norm = compute_fft_norm(v2, dt, p_factor)
    freqs_v3, amp_v3, amp_v3_norm = compute_fft_norm(v3, dt, p_factor)


    #Plotting:

    # Time excerpts for x(t)
    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], x1[short_slice], linewidth=0.9, color='C0')
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    print(f'x0 = 1 m — Time series (first {n_short} steps ≈ {time[short_slice][-1]:.4g} s)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], x2[short_slice], linewidth=0.9, color='C1')
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    print(f'x0 = xc — Time series (first {n_short} steps ≈ {time[short_slice][-1]:.4g} s)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], x3[short_slice], linewidth=0.9, color='C2')
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    print(f'x0 = 10 xc — Time series (first {n_short} steps ≈ {time[short_slice][-1]:.4g} s)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Time excerpts for v(t)
    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], v1[short_slice], linewidth=0.9, color='C0')
    plt.xlabel('Time (s)')
    plt.ylabel('v (m/s)')
    print('v(t) — x0 = 1 m (short excerpt)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], v2[short_slice], linewidth=0.9, color='C1')
    plt.xlabel('Time (s)')
    plt.ylabel('v (m/s)')
    print('v(t) — x0 = xc (short excerpt)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3.5))
    plt.plot(time[short_slice], v3[short_slice], linewidth=0.9, color='C2')
    plt.xlabel('Time (s)')
    plt.ylabel('v (m/s)')
    print('v(t) — x0 = 10 xc (short excerpt)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Normalized position spectra on same axes
    plt.figure(figsize=(9, 5))
    plt.plot(freqs_x1, amp_x1_norm, label='x(t), x0 = 1 m', linewidth=1.0)
    plt.plot(freqs_x2, amp_x2_norm, label='x(t), x0 = xc', linewidth=1.0)
    plt.plot(freqs_x3, amp_x3_norm, label='x(t), x0 = 10 xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * pi))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized amplitude |x̂(f)| / |x̂|max')
    print('Normalized position spectra')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Normalized velocity spectra on same axes
    plt.figure(figsize=(9, 5))
    plt.plot(freqs_v1, amp_v1_norm, label='v(t), x0 = 1 m', linewidth=1.0)
    plt.plot(freqs_v2, amp_v2_norm, label='v(t), x0 = xc', linewidth=1.0)
    plt.plot(freqs_v3, amp_v3_norm, label='v(t), x0 = 10 xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * pi))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized amplitude |v̂(f)| / |v̂|max')
    print('Normalized velocity spectra')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Position vs velocity comparison (same figure, two panels)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs_x1, amp_x1_norm, label='x0=1 m', linewidth=1.0)
    plt.plot(freqs_x2, amp_x2_norm, label='x0=xc', linewidth=1.0)
    plt.plot(freqs_x3, amp_x3_norm, label='x0=10 xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * pi))
    plt.ylabel('Normalized |x̂|')
    print('Position spectra (top) and velocity spectra (bottom)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(freqs_v1, amp_v1_norm, label='x0=1 m', linewidth=1.0)
    plt.plot(freqs_v2, amp_v2_norm, label='x0=xc', linewidth=1.0)
    plt.plot(freqs_v3, amp_v3_norm, label='x0=10 xc', linewidth=1.0)
    plt.xlim(0, 5.0 * omega / (2 * pi))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized |v̂|')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
