import numpy as np

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



def runge_kutta4_4D(rhs, r0, t_points, dt):
    """
    RK4 solver for 4D system r = [x, vx, y, vy] with rhs(r) returning [vx, Fx, vy, Fy].
    Parameters:
        rhs      : function rhs(r) -> np.array([vx, Fx, vy, Fy])
        r0       : initial state np.array([x0, vx0, y0, vy0])
        t_points : iterable of times where solution is recorded
        dt       : step size to use in RK4
    Returns:
        x_points, vx_points, y_points, vy_points as numpy arrays
    """
    r = np.copy(r0)
    x_points, vx_points, y_points, vy_points = [], [], [], []

    for t in t_points:
        # record current state
        x_points.append(r[0])
        vx_points.append(r[1])
        y_points.append(r[2])
        vy_points.append(r[3])

        # RK4 coefficients (rhs has no explicit t-dependence)
        k1 = dt * rhs(r)
        k2 = dt * rhs(r + 0.5 * k1)
        k3 = dt * rhs(r + 0.5 * k2)
        k4 = dt * rhs(r + k3)
        r += (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return (np.array(x_points),
            np.array(vx_points),
            np.array(y_points),
            np.array(vy_points))


# Example usage (assuming rhs(r) is defined as in your starter code)
if __name__ == "__main__":
    a = 0.0
    b = 10.0
    N = 1000
    dt = (b - a) / N
    t_points = np.arange(a, b, dt)

    r0 = np.array([1., 0., 0., 1.], float)

    x_pts, vx_pts, y_pts, vy_pts = runge_kutta4_4D(rhs, r0, t_points, dt)

    # simple check: lengths match
    print("Computed points:", len(x_pts))
