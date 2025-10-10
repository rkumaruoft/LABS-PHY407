import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, y, sigma):
    return np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))


if __name__ == "__main__":
    # read in the blur img
    with open("blur.txt", 'r') as file:
        grid = np.array([list(map(float, line.split())) for line in file if line.strip()])

    grid = np.flipud(grid)

    sigma = 25
    gaussian_arr = np.zeros_like(grid)
    nx, ny = grid.shape

    # Compute coordinates relative to center
    cx, cy = nx // 2, ny // 2

    for i in range(nx):
        for j in range(ny):
            # Shift so that (0,0) is the center of the array
            x = i - cx
            y = j - cy
            val = gaussian(x, y, sigma)

            # Assign symmetrically (avoid double-counting when i or j = 0)
            gaussian_arr[i, j] = val

    # Roll (wrap) to make it periodic
    gaussian_grid = np.roll(np.roll(gaussian_arr, cx, axis=0), cy, axis=1)

    # fourier transform the grids
    img_grid_fft = np.fft.fft2(grid)
    gauss_grid_fft = np.fft.fft2(gaussian_grid)

    restored_fft = np.zeros_like(grid, dtype=complex)
    for i in range(nx):
        for j in range(ny):
            gauss_val = gauss_grid_fft[i, j]
            img_val = img_grid_fft[i, j]

            if abs(gauss_val) > (10 ** -3):
                restored_fft[i, j] = img_val / gauss_val
            else:
                restored_fft[i, j] = img_val

    restored_grid = np.fft.ifft2(restored_fft)
    restored_grid = restored_grid.real

    # ---Show the restored image ---
    plt.imshow(restored_grid, origin='lower')
    plt.colorbar(label="Intensity")
    plt.savefig("Plots/restored.png", dpi=300, bbox_inches="tight")
    plt.show()
