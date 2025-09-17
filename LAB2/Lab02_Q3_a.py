
import numpy as np
import matplotlib.pyplot as plt
import sympy

def V(u, r, z):
    numerator = Q*np.exp(-np.tan(u)**2))
    denominator = 4*np.pi*((np.cos(u))**2)*np.sqrt((z-l*np.sqrt(u))**2 + r**2)
    return numerator/denominator
