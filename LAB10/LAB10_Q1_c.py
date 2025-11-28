"""
Lab 10
Question 1(c): Random points on the surface of the Earth
Author: Reeshav Kumar (November 2025)

Purpose:
    Compute the land fraction of Earth using numerical integration

"""
import numpy as np

if __name__ == "__main__":
    with np.load('Earth.npz') as npz:
        data = npz['data']  # (lon, lat)
        lat = npz['lat']  # degrees, length 2160
        lon = npz['lon']  # degrees, length 4320

    # convert to radians
    lat = np.radians(lat)
    lon = np.radians(lon)

    # spherical angles
    thetas = (np.pi / 2) - lat  # shape (2160,)
    phis = lon + np.pi  # shape (4320,)

    # grid spacings
    delta_theta = abs(thetas[1] - thetas[0])
    delta_phi = abs(phis[1] - phis[0])

    # calculate the area elements
    area_elements = np.sin(thetas) * delta_theta * delta_phi
    area_elements_grid = area_elements[np.newaxis, :]

    land_fraction = np.sum(area_elements_grid * data) / (4 * np.pi)

    print(f'LAND FRACTION: {land_fraction}')
