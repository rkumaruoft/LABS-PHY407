# %load throw.py # from Newman
from numpy import array, arange
g = 9.81 # Acceleration due to gravity
a = 0.0 # Initial time
b = 10.0 # Final time
N = 1000 # Number of Runge-Kutta steps
h = (b-a)/N # Size of Runge-Kutta steps
target = 1e-10 # Target accuracy for binary search

def f(r):
"""Function for Runge-Kutta calculation"""
    x = r[0]
    y = r[1]
    fx = y
    fy = -g
    return array([fx, fy], float)


