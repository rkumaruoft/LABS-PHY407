from numpy import zeros
from cmath import exp, pi
import numpy as np



# The Discrete Fourier Transform Function
def dft(y):
    N = len(y)
    c = zeros(N // 2 + 1, complex)
    for k in range(N // 2 + 1):
        for n in range(N):
            c[k] += y[n] * exp(-2j * pi * k * n / N)
    return c

def gaussxw(N):
    """Return nodes x and weights w for Legendre-Gauss on [-1,1]."""
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = np.cos(pi * a + 1.0 / (8 * N * N * np.tan(a)))
    # Newton's method
    epsilon = 1e-15
    delta = 1.0
    from numpy import ones, copy
    while delta > epsilon:
        p0 = ones(N, float)
        p1 = copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w

def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    xm = 0.5 * (b - a) * x + 0.5 * (b + a)
    wm = 0.5 * (b - a) * w
    return xm, wm

def gaussian_quad(f_callable, N_steps, a, b):
    """Gaussian quadrature on [a,b] using N_steps nodes, expects f_callable(x)."""
    if N_steps <= 0:
        raise ValueError("N_steps must be positive")
    x_nodes, w_nodes = gaussxwab(N_steps, a, b)
    I = 0.0
    # Evaluate f_callable at nodes (support vectorized or scalar)
    try:
        vals = f_callable(x_nodes)
    except Exception:
        # fallback to scalar calls
        vals = np.array([f_callable(xi) for xi in x_nodes])
    I = np.sum(w_nodes * vals)
    return I