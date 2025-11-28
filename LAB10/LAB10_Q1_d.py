"""
Lab 10
Question 1(c): Random points on the surface of the Earth
Author: Reeshav Kumar (November 2025)
Purpose: Perform a Monte Carlo simulation to estimate the land fraction
         of the Earth.

Outputs:
    - Land fraction estimates for N = 50, 500, 5000, 50000
    - A scatter plot for N = 50000 showing land points (green) and
      water points (blue) on a longitude–latitude map.
"""

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

    # List of N values
    N_values = [50, 500, 5000, 50000]

    # Loop over all N values
    for N in N_values:

        # Generate N random points uniformly on the sphere
        theta_list = []
        phi_list = []

        for _ in range(N):
            z1 = random()
            z2 = random()

            theta = np.arccos(1 - 2 * z1)  # co-latitude
            phi = 2 * np.pi * z2  # longitude  in [0, 2π]

            theta_list.append((np.pi / 2) - theta)  # convert to latitude (rad)
            phi_list.append(phi - np.pi)  # convert to longitude (rad)

        lat = np.degrees(theta_list)
        lon = np.degrees(phi_list)

        # Clip values into valid data ranges to avoid interpolation errors
        lon = np.clip(lon, np.min(data_lon), np.max(data_lon))
        lat = np.clip(lat, np.min(data_lat), np.max(data_lat))

        points = np.column_stack((lon, lat))
        data_r = interp(points)

        # Monte Carlo land fraction
        land_fraction = np.sum(data_r) / N
        print(f"Land fraction for {N} points = {land_fraction}")

        if N == 50000:
            plt.figure(figsize=(12, 6))

            # Water points as blue dots
            plt.scatter(
                lon[data_r == 0], lat[data_r == 0],
                s=1, color="blue", alpha=0.1, label="Water"
            )

            # Land points as green dots
            plt.scatter(
                lon[data_r == 1], lat[data_r == 1],
                s=1, color="green", alpha=1, label="Land"
            )

            plt.title("Monte Carlo Earth Map for N = 50000")
            plt.xlabel("Longitude (degrees)")
            plt.ylabel("Latitude (degrees)")

            # Set axis limits to match Earth coordinate ranges
            plt.xlim([-185, 185])
            plt.ylim([-95, 95])

            # Tick labels for clarity
            plt.xticks([-180, -120, -60, 0, 60, 120, 180])
            plt.yticks([-90, -60, -30, 0, 30, 60, 90])

            plt.legend(markerscale=5)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig("mote_carlo_map.png", dpi=300)
