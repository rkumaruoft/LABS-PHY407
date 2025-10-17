import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # read in the blur img
    with open("blur.txt", 'r') as file:
        grid = np.array([list(map(float, line.split())) for line in file if line.strip()])

    """
    Draws the grid as a density plot.
    """
    grid = np.flipud(grid)
    plt.imshow(grid, origin='lower')
    plt.colorbar(label='intensity')
    plt.savefig("Plots/blurred.png", dpi=300, bbox_inches="tight")
    plt.show()
