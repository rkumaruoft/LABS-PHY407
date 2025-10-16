import time
import numpy as np
from common_functions import rungekutta4, f


def energy_change(dt):
    """
    Integrate the system for a given dt and return ΔE = max(E) - min(E).
    This quantifies how well energy is conserved numerically.
    """
    a = 1e-4
    b = 60
    t_points = np.arange(a, b, dt)
    r0 = np.array([1.0, 0.0])
    x, v = rungekutta4(f, r0, t_points, dt)
    E = v ** 2 + x ** 2
    return np.max(E) - np.min(E)


if __name__ == "__main__":
    dt_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print("dt (s)\t\tΔE (Energy change)")
    for dt in dt_values:
        start_time = time.time()
        delta_E = energy_change(dt)
        elapsed = time.time() - start_time
        print(f"{dt:.1e}\t\t{delta_E:.6e}\t\t{elapsed:.3f}")
