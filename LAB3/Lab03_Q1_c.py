import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from common_funcs import *

if __name__ == '__main__':
    #Parameters
    lam = 1
    x_min, x_max = -3, 10
    z_min, z_max = 1, 5
    Nx = 256
    Nz = 256

    #Gauss quadrature points
    N_gauss = 50

    #Create grid
    x = np.linspace(x_min, x_max, Nx)
    z = np.linspace(z_min, z_max, Nz)
    X, Z = np.meshgrid(x, z)     # Z rows, X columns

    U = X * np.sqrt(2.0 / (lam * Z))

    #Arrays for C, S, and intensity
    C_grid = np.zeros_like(U)
    S_grid = np.zeros_like(U)

    for iz in range(Nz):
        u_row = U[iz, :]
        for ix, ui in enumerate(u_row):
            if ui == 0.0:
                C_grid[iz, ix] = 0.0
                S_grid[iz, ix] = 0.0
            elif ui > 0.0:
                C_grid[iz, ix] = gaussian_quad(der_C, N_gauss, 0.0, ui)
                S_grid[iz, ix] = gaussian_quad(der_S, N_gauss, 0.0, ui)
            else:
                C_grid[iz, ix] = gaussian_quad(der_C, N_gauss, ui, 0.0)
                S_grid[iz, ix] = gaussian_quad(der_S, N_gauss, ui, 0.0)

        print(f"{iz + 1} out of {Nz}")

    I_grid = (2.0 * C_grid + 1.0)**2 + (2.0 * S_grid + 1.0)**2

    #Plotting
    plt.figure(0)
    extent = [x_min, x_max, z_max, z_min]
    im = plt.imshow(I_grid, extent=extent, aspect='auto', cmap='inferno')
    plt.colorbar(im, label='I / I0')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title(f'2D intensity (N={N_gauss})')
    plt.grid(False)
    plt.tight_layout()

    plt.show()