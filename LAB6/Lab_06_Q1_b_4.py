import numpy as np

from common_functions import *
import matplotlib.pyplot as plt
from Lab_06_Q1_b_3 import get_x0


def f_u(r, t, omega, tau, gamma, v_p, v_f):
    """
    Returns derivatives [du/dt, dv/dt] for the equation:
        u'' + (1/tau) u' + omega^2 u
        = -gamma [exp(-|v_p + u'|/v_f) - exp(-|v_p|/v_f)]
    """
    u, v = r
    du_dt = v
    dv_dt = -(v / tau) - (omega ** 2) * u - gamma * (
            np.exp(-np.abs(v_p + v) / v_f) - np.exp(-np.abs(v_p) / v_f)
    )
    return np.array([du_dt, dv_dt])


def get_vp(tau, gamma, v_f):
    ratio = (gamma * tau) / v_f
    C = v_f * np.log(ratio)
    return C


if __name__ == "__main__":
    # constants
    tau = 1
    gamma = 0.5
    v_f = 0.1
    omega = 1

    v_p = get_vp(tau, gamma, v_f)
    x0 = get_x0(omega, v_p, tau, v_f, gamma)

    # initial condition for u and u'
    u0 = 0.1
    v0 = -0.0001
    r0 = np.array([u0, v0])

    # time setup
    a = 1e-4
    b = 20
    dt = 1e-3
    t_points = np.arange(a, b, dt)

    # integrate u-equation using RK4
    u_points, v_points = runge_kutta4(
        lambda r, t: f_u(r, t, omega, tau, gamma, v_p, v_f), r0, t_points, dt
    )

    # total x(t) reconstruction
    x_points = x0 + (v_p * t_points) + u_points

    plt.figure(figsize=(8, 5))
    plt.plot(t_points, u_points, label="u(t) — perturbation")
    plt.plot(t_points, x_points, label="x(t) = x₀ + vₚt + u(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ut.png", dpi=300, bbox_inches="tight")
    plt.show()
