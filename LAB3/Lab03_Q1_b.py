import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from common_funcs import *

def der_C(t):
    return np.cos(0.5 * np.pi * t**2)

def der_S(t):
    return np.sin(0.5 * np.pi * t**2)

if __name__ == '__main__':
    #Parameters
    lam = 1.0
    z = 3.0
    x_min, x_max = -5.0, 5.0
    Nx = 1001
    N_gauss = 50

    x = np.linspace(x_min, x_max, Nx)
    u = x * np.sqrt(2.0 / (lam * z))

    #Computing Fresnel integrals using Gaussian quadrature per u
    C_g = np.zeros_like(u)
    S_g = np.zeros_like(u)
    for i, ui in enumerate(u):
        if ui == 0.0:
            C_g[i] = 0.0
            S_g[i] = 0.0
        elif ui > 0.0:
            C_g[i] = gaussian_quad(der_C, N_gauss, 0.0, ui)
            S_g[i] = gaussian_quad(der_S, N_gauss, 0.0, ui)
        else:
            C_g[i] = gaussian_quad(der_C, N_gauss, ui, 0.0)
            S_g[i] = gaussian_quad(der_S, N_gauss, ui, 0.0)
        if i % 100 == 0:
            print(i)
    #Intensity ratio I/I0
    I_over_I0 = (2.0 * C_g + 1.0)**2 + (2.0 * S_g + 1.0)**2

    #SciPy reference
    S_scipy, C_scipy = sc.fresnel(u)
    I_over_I0_scipy = (2.0 * C_scipy + 1.0)**2 + (2.0 * S_scipy + 1.0)**2

    #Relative difference
    rel_err = get_fraction_err(I_over_I0, I_over_I0_scipy)

    #Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(x, I_over_I0_scipy, label='I/I0 (SciPy)', color='C0', lw=1.6)
    plt.plot(x, I_over_I0, '--', label=f'I/I0 (Gauss N={N_gauss})', color='C1', lw=1.0)
    plt.ylabel('I / I0')
    plt.title('Near-field diffraction by a straight edge (λ=1 m, z=3 m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(9, 3))
    plt.semilogy(x, rel_err, color='C2')
    plt.xlabel('x (m)')
    plt.ylabel('relative difference')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()


