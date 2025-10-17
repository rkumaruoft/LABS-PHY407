import numpy as np


def g(r, t, omega, tau, gamma, v_p, v_f):
    """
    Computes acceleration (dv/dt).
    Parameters:
        r     : array([x, v])  → position and velocity
        t     : float          → time
        omega : float          → oscillation frequency (sqrt(k/m))
        tau   : float          → fluid damping time-constant
        gamma : float          → static friction coefficient
        v_p   : float          → pulling velocity of spring end
        v_f   : float          → velocity threshold for static friction

    Returns:
        float : acceleration dv/dt
    """
    x, v = r
    return -(omega ** 2) * (x - v_p * t) - (v / tau) - gamma * np.exp(-np.abs(v) / v_f)


def f(r, t, omega=1, tau=1e13, gamma=0, v_p=0, v_f=5):
    """
    Returns time derivatives of [x, v] for RK4 integration.

    dx/dt = v
    dv/dt = g(r, t, ...)

    Parameters:
        r : np.array([x, v])
        t : time (float)
        omega : float          → oscillation frequency (sqrt(k/m))
        tau   : float          → fluid damping time-constant
        gamma : float          → static friction coefficient
        v_p   : float          → pulling velocity of spring end
        v_f   : float          → velocity threshold for static friction

    Returns:
        np.array([dx/dt, dv/dt])
    """
    x, v = r
    fx = v
    fv = g(r, t, omega, tau, gamma, v_p, v_f)
    return np.array([fx, fv])


def runge_kutta4(f, r0, t_points, dt):
    """
    General RK4 solver for 2D systems of form:
        dx/dt = v
        dv/dt = f(x, v, t)

    Parameters:
        f         : derivative function f(r, t) returning [dx/dt, dv/dt]
        r0        : np.array([x0, v0])  initial conditions
        t_points  : np.array of time points
        dt        : step size (float)

    Returns:
        x_points  : np.array of x(t)
        v_points  : np.array of v(t)
    """
    r = np.copy(r0)
    x_points, v_points = [], []

    for t in t_points:
        x_points.append(r[0])
        v_points.append(r[1])

        k1 = dt * f(r, t)
        k2 = dt * f(r + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * f(r + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * f(r + k3, t + dt)
        r += (k1 + 2*k2 + 2*k3 + k4) / 6

    return np.array(x_points), np.array(v_points)