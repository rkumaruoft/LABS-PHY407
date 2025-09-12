

import numpy as np
import matplotlib.pyplot as plt

M_s = 1
M_p = 1.6e-7
G = 39.5
alpha = 0.01


def get_ACC(G, M_s, r):
    return (G * M_s) / (r ** 3)


def get_acc_relativity(G, M_s, r, alpha):
    return ((G * M_s) / (r ** 3)) * (1 + (alpha / (r ** 2)))


def get_radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def func_x(x_i, vx_i, r, dt):
    return x_i + (vx_i * dt), vx_i - (get_ACC(G, M_s, r) * x_i * dt)


def func_y(y_i, vy_i, r, dt):
    return y_i + (vy_i * dt), vy_i - (get_ACC(G, M_s, r) * y_i * dt)


def func_x_relativity(x_i, vx_i, r, dt):
    return x_i + (vx_i * dt), vx_i - (get_acc_relativity(G, M_s, r, alpha) * x_i * dt)


def func_y_relativity(y_i, vy_i, r, dt):
    return y_i + (vy_i * dt), vy_i - (get_acc_relativity(G, M_s, r, alpha) * y_i * dt)


if __name__ == "__main__":
    """
    Initial conditions
    """
    # time in years
    dt = 0.0001
    total_time = 1
    n = int(total_time / dt)
    t = np.arange(n) * dt

    # initialize the arrays
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    speed_n = np.zeros(n)

    x[0] = 0.47
    y[0] = 0
    vx[0] = 0
    vy[0] = 8.17

    for i in range(n - 1):
        r = get_radius(x[i], y[i])
        x[i + 1], vx[i + 1] = func_x(x[i], vx[i], r, dt)
        y[i + 1], vy[i + 1] = func_y(y[i], vy[i], r, dt)
        speed_n[i] = np.hypot(vx[i], vy[i])

    speed_n[-1] = np.hypot(vx[-1], vy[-1])

    plt.figure(1)
    plt.xlabel("x(AU)")
    plt.ylabel("y(AU)")
    plt.plot(np.array(x), np.array(y), label="Mercury")
    plt.scatter(0, 0, color="gold", s=100, label="Sun")
    plt.axis("equal")
    plt.legend()
    plt.show()

    # part 2 with relativity

    # initialize the arrays
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    speed_r = np.zeros(n)

    x[0] = 0.47
    y[0] = 0
    vx[0] = 0
    vy[0] = 8.17

    for i in range(n - 1):
        r = get_radius(x[i], y[i])
        x[i + 1], vx[i + 1] = func_x_relativity(x[i], vx[i], r, dt)
        y[i + 1], vy[i + 1] = func_y_relativity(y[i], vy[i], r, dt)
        speed_r[i] = np.hypot(vx[i], vy[i])

    speed_r[-1] = np.hypot(vx[-1], vy[-1])

    plt.figure(2)
    plt.xlabel("x(AU)")
    plt.ylabel("y(AU)")
    plt.plot(np.array(x), np.array(y), label="Mercury")
    plt.scatter(0, 0, color="gold", s=100, label="Sun")
    plt.axis("equal")
    plt.legend()
    plt.show()

    #Velocity vs. time plots
    plt.figure(3, figsize=(8, 4))
    plt.plot(t, speed_n, label="Newtonian", linewidth=1)
    plt.plot(t, speed_r, '--', label="Relativistic", linewidth=1)
    plt.xlabel("Time (years)")
    plt.ylabel("Velocity (AU/year)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
