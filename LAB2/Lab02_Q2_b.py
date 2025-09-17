import numpy as np
from Lab02_Q2_a import simpson_int


def differential_bessel(x, level, angle):
    return (1 / np.pi) * np.cos(level * angle - x * np.sin(angle))


def bessel_integrand(x, level):
    return lambda angle: differential_bessel(x, level, angle)


if __name__ == "__main__":
    N_steps = 1000

    # For J_0
    level = 0

    bessel_j0 = []
    for x in range(0, 20):
        bessel_j0.append(simpson_int(bessel_integrand(x, 0), N_steps, lower_limit=0, upper_limit=np.pi))

    print(bessel_j0)

