import numpy as np
import matplotlib.pyplot as plt

M_s = 1
G = 39.5
alpha = 0.01

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

    plt.figure(1)
    plt.xlabel("x(AU)")
    plt.ylabel("y(AU)")
    plt.plot(np.array(x), np.array(y))
    plt.show()
