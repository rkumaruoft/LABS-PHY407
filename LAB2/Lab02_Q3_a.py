
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k0 as scipy_modified_bessel

def potential(u, r, z):
    l=1e-3
    Q=10e-13
    eps = 8.854e-12
    numerator = Q*np.exp(-np.tan(u)**2)
    denominator = 4*np.pi*eps*((np.cos(u))**2)*np.sqrt((z-l*np.tan(u))**2 + r**2)
    return numerator/denominator

def potential_soln(r, z=0):
    l = 1e-3
    Q = 10e-13
    eps = 8.854e-12
    arg = (r**2) / (2 * l**2)
    val = (Q/(4*np.pi*eps*l)) * np.exp(arg) * scipy_modified_bessel(arg)
    return float(val)


def potential_integrand(r, z):
    return lambda u: potential(u, r, z)

def simpson_int(diff_func, N_steps, lower_limit, upper_limit):
    h = (upper_limit - lower_limit) / N_steps  # width of slice

    sum1 = 0
    for k in range(1, N_steps, 2):
        sum1 += diff_func(lower_limit + k * h)

    sum2 = 0
    for k in range(2, N_steps, 2):
        sum2 += diff_func(lower_limit + k * h)

    s = (h / 3) * (diff_func(lower_limit) + diff_func(upper_limit) +
                   4 * sum1 + 2 * sum2)
    return s

#integration limits
a, b = -np.pi/2, np.pi/2

#r-range:
start, stop, step = 0.25e-3, 5e-3, 0.05e-3
r_array = np.arange(start, stop + step/2, step)

#Number of subintervals
N_steps = 50  # must be even

x_vals = (0.25**2)/(2*1**2)

simpson_vals = []
exact_vals   = []
frac_errors = []

for r in r_array:
    V_sim = simpson_int(potential_integrand(r, 0), N_steps, a, b)
    V_ex  = potential_soln(r, 0)

    simpson_vals.append(V_sim)
    exact_vals.append(V_ex)

    rel_err = (V_sim - V_ex) / V_ex
    print(f"r = {r:.2e}  |  V_simpson = {V_sim:.4e}  |  V_exact = {V_ex:.4e}  |  rel_error = {rel_err:.2e}")

    frac_errors.append(abs((rel_err / V_sim)) * 100)

print("The average fractional error is: ", np.mean(frac_errors), "%")

#Plotting
plt.figure(figsize=(8, 5))
plt.plot(r_array*1e3, simpson_vals, label="Simpson’s rule")
plt.plot(r_array*1e3, exact_vals,   '--', label="Analytic (K0)")
plt.xlabel("r (mm)")
plt.ylabel("V (V)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(r_array*1e3, (np.array(simpson_vals)-np.array(exact_vals))/np.array(exact_vals))
plt.xlabel("r (mm)")
plt.ylabel("Relative error")
plt.tight_layout()
plt.show()
