import numpy as np
import matplotlib.pyplot as plt


def intensity_transmission_func(u):
    return np.sin(np.pi * u / 20e-6) ** 2


if __name__ == "__main__":
    # constants
    w = 200e-6  # m
    W = 10 * w  # m
    wavelength = 500e-9  # m
    focal_length = 1  # m
    N = 10000

    # define u range
    u_range = np.linspace(-W / 2, W / 2, N)
    y = np.sqrt(np.sin(np.pi * u_range / 20e-6) ** 2)
    y[np.abs(u_range) > w / 2] = 0

    # FFT
    c_k = np.fft.fft(y)
    k = np.fft.fftfreq(N, d=W / N)
    x_k = (wavelength * focal_length / W) * k

    I = (W ** 2 / N ** 2) * np.abs(c_k) ** 2
    I = I / np.max(I)

    # plot like reference code
    plt.scatter(x_k, I, marker='.', linewidth=1)
    plt.xlim(-5, 5)
    plt.xlabel('x (m)')
    plt.ylabel('Intensity')
    plt.title('Diffraction Intensity Pattern')
    plt.savefig("Plots/diffraction_pattern.png", dpi=300, bbox_inches="tight")
    plt.grid(True)
    plt.show()
