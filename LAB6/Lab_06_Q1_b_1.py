import numpy as np
import matplotlib.pyplot as plt
from common_functions import runge_kutta4, f


if __name__ == "__main__":

    a = 1e-4        # start time (s)
    b = 60          # end time (s)
    dt = 0.005      # time step (s)
    t_points = np.arange(a, b, dt)
    r0 = np.array([1.0, 0.0])  # initial conditions [x0, v0]

    # these conditions plot around 10 oscillations
    x_points, v_points = runge_kutta4(f, r0, t_points, dt)


    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(t_points, x_points, label="x(t) — Displacement")
    plt.plot(t_points, v_points, label="v(t) — Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
