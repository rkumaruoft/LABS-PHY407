import numpy as np

from common_functions import *
import matplotlib.pyplot as plt


def get_x0(omega, c, tau, v_f, gamma):
    return (-1 / (omega ** 2)) * ((c / tau) + (gamma * np.exp(-np.abs(c) / v_f)))


if __name__ == "__main__":
    # constants
    tau = 1
    gamma = 0.5
    v_f = 0.1
    omega = 1

    # guess v_p
    v_p = 1

    # constant solution
    x0 = get_x0(omega, v_p, tau, v_f, gamma)
    r0 = np.array([x0, v_p])

    f_custom = lambda r, t: f(r, t, omega=omega, tau=tau, gamma=gamma, v_p=v_p, v_f=v_f)

    a = 1e-4  # start time (s)
    b = 20  # end time (s)
    dt = 1e-3  # time step (s)
    t_points = np.arange(a, b, dt)

    # Integrate using RK4
    x_points, v_points = runge_kutta4(f_custom, r0, t_points, dt)



    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(t_points, x_points, label="x(t) — Displacement")
    plt.plot(t_points, v_points, label="v(t) — Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
