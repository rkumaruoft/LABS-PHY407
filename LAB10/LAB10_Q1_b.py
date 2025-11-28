"""
Lab 10
Question 1: Random points on the surface of the Earth
Author: Reeshav Kumar (November 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

if __name__ == "__main__":

    seed(12345)

    N = 5000
    theta_list = []
    phi_list = []
    xyz_list = []

    # Generate random points
    for i in range(N):
        z1 = random()
        z2 = random()

        theta = np.arccos(1 - 2 * z1)
        phi = 2 * np.pi * z2

        theta_list.append((np.pi / 2) - theta)
        phi_list.append(phi - np.pi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        xyz_list.append((x, y, z))

    theta_list = np.array(theta_list)
    phi_list = np.array(phi_list)
    xyz_list = np.array(xyz_list)

    lat = np.array(np.degrees(theta_list))
    lon = np.array(np.degrees(phi_list))

    # 2D Plot: theta vs phi
    plt.figure()
    plt.scatter(lon, lat, s=1)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Random Points on Earth (Lon vs Lat)")

    # Set axis limits
    plt.xlim([-185, 185])
    plt.ylim([-95, 95])

    plt.xticks([-180, -120, -60, 0, 60, 120, 180])
    plt.yticks([-90, -60, -30, 0, 30, 60, 90])

    # Add gridlines to show boundaries clearly
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("theta_phi_plot.png", dpi=300)
    plt.close()

    # 3D Plot: Cartesian scatter
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz_list[:, 0], xyz_list[:, 1], xyz_list[:, 2], s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Random Points on Earth (x, y, z)", pad=15)

    plt.savefig("cartesian_plot.png", dpi=200)
    plt.close()
