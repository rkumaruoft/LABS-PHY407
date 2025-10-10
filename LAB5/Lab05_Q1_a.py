import numpy as np
import matplotlib.pyplot as plt
import os
from common_funcs import dft

# define wave functions
def square_wave(x, L):
    x_mod = x % L
    return np.where(x_mod >= L / 2, 1, 0)  # 0 in first half, 1 in second half

def sawtooth_wave(x, L):
    x_mod = x % L
    return x_mod / L  # linear ramp 0 → 1 each period

def modulated_sine_wave(x, L):
    return np.sin(np.pi * x / L) * np.sin(20 * np.pi * x / L)  # envelope * carrier

# create folder if not exists
os.makedirs("Plots", exist_ok=True)

if __name__ == "__main__":
    # square wave
    square_x = np.linspace(0, 1, 1000, endpoint=False)
    square_y = square_wave(square_x, 1)

    plt.figure()
    plt.plot(square_x, square_y, linewidth=2, label="Square Wave")  # time-domain
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Plots/square_wave.png", dpi=300, bbox_inches="tight")
    plt.close()

    fourier_coeffs = dft(square_y)  # DFT
    n = np.arange(len(fourier_coeffs))
    amplitudes = np.abs(fourier_coeffs)

    plt.figure()
    plt.bar(n[:100], amplitudes[:100], label="Square Wave Spectrum")
    plt.xlabel("Frequency Index (n)")
    plt.ylabel("Amplitude $|C_n|$")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Plots/square_wave_dft.png", dpi=300, bbox_inches="tight")
    plt.close()

    # sawtooth wave
    saw_x = np.linspace(0, 2, 1000, endpoint=False)
    saw_y = sawtooth_wave(saw_x, 1)

    plt.figure()
    plt.plot(saw_x, saw_y, linewidth=2, label="Sawtooth Wave")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Plots/sawtooth_wave.png", dpi=300, bbox_inches="tight")
    plt.close()

    fourier_coeffs_saw = dft(saw_y)
    n = np.arange(len(fourier_coeffs_saw))
    amplitudes = np.abs(fourier_coeffs_saw)

    plt.figure()
    plt.bar(n[:100], amplitudes[:100], label="Sawtooth Wave Spectrum")
    plt.xlabel("Frequency Index (n)")
    plt.ylabel("Amplitude $|C_n|$")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Plots/sawtooth_wave_dft.png", dpi=300, bbox_inches="tight")
    plt.close()

    # modulated sine wave
    sin_x = np.linspace(0, 1, 1000, endpoint=False)
    sin_y = modulated_sine_wave(sin_x, 1)

    plt.figure()
    plt.plot(sin_x, sin_y, linewidth=2, label="Modulated Sine Wave")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Plots/modulated_sine_wave.png", dpi=300, bbox_inches="tight")
    plt.close()

    fourier_coeffs_sin = dft(sin_y)
    n = np.arange(len(fourier_coeffs_sin))
    amplitudes = np.abs(fourier_coeffs_sin)

    plt.figure()
    plt.bar(n[10:30], amplitudes[10:30], label="Modulated Sine Spectrum")  # zoomed view
    plt.xlabel("Frequency Index (n)")
    plt.ylabel("Amplitude $|C_n|$")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Plots/modulated_sine_wave_dft.png", dpi=300, bbox_inches="tight")
    plt.close()
