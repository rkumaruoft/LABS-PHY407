"""
Lab 10
Question 2: Limb darkening of a star
Purpose: perform a Monte Carlo simulation to determine the final scattering angle of the photon as it
leaves the photosphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

loaded = np.load('Earth.npz')
data = loaded['data']
lon_array = loaded['lon']
lat_array = loaded['lat']

def get_tau_step():
    """ Calculate how far a photon travels before it gets scattered.
    OUT: optical depth traveled """
    delta_tau = -np.log(np.random.random())
    return delta_tau

def emit_photon(tau_max):
    """ Emit a photon from the stellar core.
    IN: tau_max is max optical depth
    OUT:
    tau: optical depth at which the photon is created
    mu: directional cosine of the photon emitted """
    tau = tau_max
    delta_tau = get_tau_step()
    mu = np.random.random()
    return tau - delta_tau*mu, mu

def scatter_photon(tau):
    """ Scatter a photon.
    IN: tau, optical depth of the atmosphere
    OUT:
        tau: new optical depth
        mu: directional cosine of the photon scattered """

    delta_tau = get_tau_step()
    mu = 2 * np.random.random() - 1  # sample mu uniformly from -1 to 1
    return tau - delta_tau * mu, mu

if __name__ == "__main__":
    X = np.arange(0., 2*np.pi, 2*np.pi/5) # from 0 to 2pi (excluded), with 5 points
    Y = np.linspace(0., np.pi, 3) # from 0 to pi (included) with 3 points
    Yg, Xg = np.meshgrid(Y, X) # 2D grids to create data wit correct shape
    data = np.sin(Xg*Yg) # creating some data
    # create nearest interpolator
    interp = RegularGridInterpolator((X, Y), data, method='nearest')
    x, y = 1., 3. # these coordinates do not fall on a grid point above
    nearest_value = interp([x, y])


    print(nearest_value, data[1, 2]) # in this example data[1, 2] is the nearest
    plt.scatter(Xg, Yg, c=data)
    plt.colorbar()
    plt.show()