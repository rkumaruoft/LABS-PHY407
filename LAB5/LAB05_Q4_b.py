import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, y, sigma):
    return np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))


if __name__ == "__main__":
    # Read in the blur image
    with open("blur.txt", 'r') as file:
        grid = np.array([list(map(float, line.split())) for line in file if line.strip()])

    nx, ny = grid.shape
    sigma = 25
    gaussian_arr = np.zeros_like(grid)

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
    gaussian_periodic = np.roll(np.roll(gaussian_arr, cx, axis=0), cy, axis=1)

    # Plot the periodic Gaussian
    plt.imshow(gaussian_periodic, origin='lower')
    plt.colorbar(label="Amplitude")
    plt.savefig("Plots/gaussian.png", dpi=300, bbox_inches="tight")
    plt.show()
