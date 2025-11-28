import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from scipy.interpolate import RegularGridInterpolator

if __name__ == "__main__":
    seed(12345)

    # Load Earth dataset
    with np.load("Earth.npz") as npz:
        data = npz["data"]
        data_lon = npz["lon"]
        data_lat = npz["lat"]

    # Build interpolator
    interp = RegularGridInterpolator(
        (data_lon, data_lat),
        data,
        method="nearest"
    )

    # N values to test
    N_values = [50, 500, 5000, 50000]

    for N in N_values:

        # Generate N random points on sphere
        theta_list = []
        phi_list = []

        for _ in range(N):
            z1 = random()
            z2 = random()

            theta = np.arccos(1 - 2 * z1)
            phi = 2 * np.pi * z2

            theta_list.append((np.pi / 2) - theta)
            phi_list.append(phi - np.pi)

        lat = np.degrees(theta_list)
        lon = np.degrees(phi_list)

        # Clip to valid range
        lon = np.clip(lon, np.min(data_lon), np.max(data_lon))
        lat = np.clip(lat, np.min(data_lat), np.max(data_lat))

        # Interpolate land/water classification
        points = np.column_stack((lon, lat))
        data_r = interp(points)

        land_fraction = np.mean(data_r)
        print(f"Land fraction for {N} points = {land_fraction}")

        # If N = 50000, plot the map
        if N == 50000:
            plt.figure(figsize=(12, 6))

            plt.scatter(lon[data_r == 0], lat[data_r == 0],
                        s=1, color="blue", alpha=0.2, label="Water")

            plt.scatter(lon[data_r == 1], lat[data_r == 1],
                        s=1, color="green", alpha=1.0, label="Land")

            plt.title("Monte Carlo Earth Map for N = 50000")
            plt.xlabel("Longitude (degrees)")
            plt.ylabel("Latitude (degrees)")
            # Set axis limits
            plt.xlim([-185, 185])
            plt.ylim([-95, 95])

            plt.xticks([-180, -120, -60, 0, 60, 120, 180])
            plt.yticks([-90, -60, -30, 0, 30, 60, 90])

            plt.legend(markerscale=5)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.show()
