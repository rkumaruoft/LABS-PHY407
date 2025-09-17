import numpy as np
import matplotlib.pyplot as plt
from Lab02_Q2_a import simpson_int
from scipy.special import jv


def differential_bessel(x, level, angle):
    return (1 / np.pi) * np.cos(level * angle - x * np.sin(angle))


def bessel_integrand(x, level):
    return lambda angle: differential_bessel(x, level, angle)


if __name__ == "__main__":
    N_steps = 1000

    bessel_j0 = []
    bessel_j3 = []
    bessel_j5 = []

    # x values from 0 to 20 with step 0.5
    x_vals = np.arange(0, 20)

    for i in x_vals:
        bessel_j0.append(simpson_int(bessel_integrand(i, 0), N_steps, lower_limit=0, upper_limit=np.pi))
        bessel_j3.append(simpson_int(bessel_integrand(i, 3), N_steps, lower_limit=0, upper_limit=np.pi))
        bessel_j5.append(simpson_int(bessel_integrand(i, 5), N_steps, lower_limit=0, upper_limit=np.pi))

    # plot bessel functions
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, bessel_j0, label=r"$J_0(x)$", color="red")
    plt.plot(x_vals, bessel_j3, label=r"$J_3(x)$", color="blue")
    plt.plot(x_vals, bessel_j5, label=r"$J_5(x)$", color="green")

    plt.xlabel("x")
    plt.ylabel(r"$J_n(x)$")
    plt.legend()
    plt.show()

    # compute Bessel functions directly
    bessel_jv0 = jv(0, x_vals)
    bessel_jv3 = jv(3, x_vals)
    bessel_jv5 = jv(5, x_vals)

    # === Plot J0 vs JV0 ===
    plt.plot(x_vals, bessel_jv0, "r--", label=r"$J_0(x)$ SciPy")
    plt.plot(x_vals, bessel_j0, "ro", label=r"$J_0(x)$ Simpson", markersize=4)
    plt.xlabel("x")
    plt.ylabel(r"$J_0(x)$")
    plt.legend()
    plt.show()

    # === Plot J3 vs JV3 ===
    plt.plot(x_vals, bessel_jv3, "b--", label=r"$J_3(x)$ SciPy")
    plt.plot(x_vals, bessel_j3, "bs", label=r"$J_3(x)$ Simpson", markersize=4)
    plt.xlabel("x")
    plt.ylabel(r"$J_3(x)$")
    plt.legend()
    plt.show()

    # === Plot J5 vs JV5 ===
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, bessel_jv5, "g--", label=r"$J_5(x)$ SciPy")
    plt.plot(x_vals, bessel_j5, "g^", label=r"$J_5(x)$ Simpson", markersize=4)
    plt.xlabel("x")
    plt.ylabel(r"$J_5(x)$")
    plt.legend()
    plt.show()

    # === Residual plots (scatter of Simpson vs SciPy) ===
    diff_J0 = np.array(bessel_j0) - bessel_jv0
    diff_J3 = np.array(bessel_j3) - bessel_jv3
    diff_J5 = np.array(bessel_j5) - bessel_jv5

    # J0 residuals
    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='black', linewidth=1)  # reference line
    plt.scatter(x_vals, diff_J0, color='red', s=15, label=r"$J_0$ residuals")
    plt.xlabel("x")
    plt.ylabel("Residual (Simpson - SciPy)")
    plt.legend()
    plt.show()

    # J3 residuals
    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(x_vals, diff_J3, color='blue', s=15, label=r"$J_3$ residuals")
    plt.xlabel("x")
    plt.ylabel("Residual (Simpson - SciPy)")
    plt.legend()
    plt.show()

    # J5 residuals
    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(x_vals, diff_J5, color='green', s=15, label=r"$J_5$ residuals")
    plt.xlabel("x")
    plt.ylabel("Residual (Simpson - SciPy)")
    plt.legend()
    plt.show()

