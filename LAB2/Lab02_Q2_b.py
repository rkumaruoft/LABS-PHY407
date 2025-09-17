import numpy as np
import matplotlib.pyplot as plt
from Lab02_Q2_a import simpson_int


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
    x_vals = np.arange(0, 20, 0.1)

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
    plt.grid(True)
    plt.show()
