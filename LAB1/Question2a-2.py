import numpy as np
import matplotlib.pyplot as plt

M_s = 1
G = 39.5

def get_ACC(G, M_s, r):
    return (G * M_s) / (r ** 3)

def get_radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def func_x(x_i, vx_i, r, dt):
    return x_i + (vx_i * dt), vx_i - (get_ACC(G, M_s, r) * x_i * dt)


def func_y(y_i, vy_i, r, dt):
    return y_i + (vy_i * dt), vy_i - (get_ACC(G, M_s, r) * y_i * dt)


if __name__ == "__main__":
    """
    Initial conditions
    """
    # time in years
    dt = 0.0001
    total_time = 10
    n = int(total_time / dt)

    # Jupiter
    # initialize the arrays
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    x[0] = 5.2
    y[0] = 0
    vx[0] = 0
    vy[0] = 2.63

    for i in range(n - 1):
        r = get_radius(x[i], y[i])
        x[i + 1], vx[i + 1] = func_x(x[i], vx[i], r, dt)
        y[i + 1], vy[i + 1] = func_y(y[i], vy[i], r, dt)


    #Earth
    # initialize the arrays
    x_e = np.zeros(n)
    y_e = np.zeros(n)
    vx_e = np.zeros(n)
    vy_e = np.zeros(n)

    x_e[0] = 1.0
    y_e[0] = 0
    vx_e[0] = 0
    vy_e[0] = 6.18

    for i in range(n - 1):
        r = get_radius(x[i], y[i])
        x_e[i + 1], vx_e[i + 1] = func_x(x_e[i], vx_e[i], r, dt)
        y_e[i + 1], vy_e[i + 1] = func_y(y_e[i], vy_e[i], r, dt)

    plt.figure(1)
    plt.xlabel("x(AU)")
    plt.ylabel("y(AU)")
    plt.plot(np.array(x), np.array(y))
    plt.show()