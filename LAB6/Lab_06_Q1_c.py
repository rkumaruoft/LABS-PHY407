import numpy as np
import matplotlib.pyplot as plt
import os
from common_functions import *

def get_x0(omega, c, tau, v_f, gamma):
    return (-1 / (omega ** 2)) * ((c / tau) + (gamma * np.exp(-np.abs(c) / v_f)))

if __name__ == "__main__":
    # constants
    tau = 1
    gamma = 0.5
    v_f = 0.1
    omega = 1

    # range for v_p as described in the question
    vp_min = 0.1 * v_f * np.log(gamma * tau / v_f)
    vp_max = 1.5 * v_f * np.log(gamma * tau / v_f)
    vp_values = np.linspace(vp_min, vp_max, 6)  # run 4 representative values

    # integration setup
    a = 1e-4
    b = 60
    dt = 1e-3
    t_points = np.arange(a, b, dt)
    r0 = np.array([0.0, 0.0])  # initial position and velocity

    # ensure folder exists
    os.makedirs("plots", exist_ok=True)

    # ---------- PLOT x(t) ----------
    plt.figure(figsize=(8, 5))
    for vp in vp_values:
        f_custom = lambda r, t: f(r, t, omega=omega, tau=tau,
                                  gamma=gamma, v_p=vp, v_f=v_f)
        x_points, v_points = runge_kutta4(f_custom, r0, t_points, dt)
        plt.plot(t_points, x_points, label=fr"$v_p = {vp:.3f}$")

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement $x(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/xt_vary_vp.png", dpi=300, bbox_inches="tight")
    plt.show()