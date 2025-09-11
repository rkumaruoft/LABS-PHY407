import numpy as np
import matplotlib.pyplot as plt

M_s = 1
G = 39.5
M_j = 0.0009543
M_e   = 0.000003003

def get_ACC(G, M, r):
    return (G * M) / (r ** 3)

def get_radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def func_x(x_i, vx_i, acc_x, dt):
    return x_i + vx_i * dt, vx_i + acc_x * dt

def func_y(y_i, vy_i, acc_y, dt):
    return y_i + vy_i * dt, vy_i + acc_y * dt


if __name__ == "__main__":
    """
    Initial conditions
    """
    # time in years
    dt = 0.0001
    total_time = 10
    n = int(total_time / dt)

    #Jupiter initialize the arrays
    x_j = np.zeros(n)
    y_j = np.zeros(n)
    vx_j = np.zeros(n)
    vy_j = np.zeros(n)

    x_j[0] = 5.2
    y_j[0] = 0
    vx_j[0] = 0
    vy_j[0] = 2.63

    #Earth initialize the arrays
    x_e = np.zeros(n)
    y_e = np.zeros(n)
    vx_e = np.zeros(n)
    vy_e = np.zeros(n)

    x_e[0] = 1.0
    y_e[0] = 0
    vx_e[0] = 0
    vy_e[0] = 6.18

    #Jupiter calculations


    for i in range(n - 1):
        #Jupiter–Sun
        r_js = get_radius(x_j[i], y_j[i])
        a_js_x = -get_ACC(G, M_s, r_js) * x_j[i]
        a_js_y = -get_ACC(G, M_s, r_js) * y_j[i]
        #Jupiter–Earth
        dx_je = x_j[i] - x_e[i]
        dy_je = y_j[i] - y_e[i]
        r_je = get_radius(dx_je, dy_je)
        a_je_x = -get_ACC(G, M_e, r_je) * dx_je
        a_je_y = -get_ACC(G, M_e, r_je) * dy_je
        #total Jupiter accel
        a_jx = a_js_x + a_je_x
        a_jy = a_js_y + a_je_y

        #update Jupiter
        x_j[i + 1], vx_j[i + 1] = func_x(x_j[i], vx_j[i], a_jx, dt)
        y_j[i + 1], vy_j[i + 1] = func_y(y_j[i], vy_j[i], a_jy, dt)

        #Earth calculations
        #Earth–Sun
        r_es = get_radius(x_e[i], y_e[i])
        a_es_x = -get_ACC(G, M_s, r_es) * x_e[i]
        a_es_y = -get_ACC(G, M_s, r_es) * y_e[i]

        #Earth–Jupiter
        dx_ej = x_e[i] - x_j[i]
        dy_ej = y_e[i] - y_j[i]
        r_ej = get_radius(dx_ej, dy_ej)
        a_ej_x = -get_ACC(G, M_j, r_ej) * dx_ej
        a_ej_y = -get_ACC(G, M_j, r_ej) * dy_ej

        #total Earth accel
        a_ex = a_es_x + a_ej_x
        a_ey = a_es_y + a_ej_y

        #update Earth
        x_e[i + 1], vx_e[i + 1] = func_x(x_e[i], vx_e[i], a_ex, dt)
        y_e[i + 1], vy_e[i + 1] = func_y(y_e[i], vy_e[i], a_ey, dt)


    #plotting orbits
    plt.figure(1)
    plt.xlabel("x(AU)")
    plt.ylabel("y(AU)")
    plt.plot(x_e, y_e, label="Earth")
    plt.plot(x_j, y_j, label="Jupiter")
    plt.scatter(0, 0, color="gold", s=100, label="Sun")
    plt.plot(np.array(x_e), np.array(y_e))
    plt.plot(np.array(x_j), np.array(y_j))
    plt.axis("equal")
    plt.legend()
    plt.show()