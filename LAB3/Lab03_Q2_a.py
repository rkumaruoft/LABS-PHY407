from common_funcs import *
import scipy.constants as consts


def spring_v_func(x, x_0, mass, spring_const):
    c = consts.c
    val_1 = spring_const * ((x_0 ** 2) - (x ** 2))
    val_2 = mass * (c ** 2)
    numerator = val_1 * ((2 * val_2) + (val_1 / 2))
    denominator = 2 * ((val_2 + (val_1 / 2)) ** 2)
    return c * np.sqrt(numerator / denominator)


def diff_period(x, x_0, mass, spring_const):
    return 4 / spring_v_func(x, x_0, mass, spring_const)


def period_integrand(x_0, mass, spring_const):
    return lambda x: diff_period(x, x_0, mass, spring_const)


if __name__ == "__main__":
    # given consts
    mass = 1  # kg
    spring_const = 12  # N/m
    boring_val = 2 * np.pi * (np.sqrt(mass / spring_const))

    # steps
    N_1 = 8
    N_2 = 16

    # case 1 x_0 = 1cm N_1 = 8
    x_0 = 1e-2
    x_vals_1, weights_1 = gaussxwab(N_1, 0, x_0)
    N_1_result = gaussian_quad(period_integrand(x_0, mass, spring_const), N_1, 0, x_0)

    # case 2 x_0 = 1cm N_2 = 16
    x_0 = 1e-2
    x_vals_2, weights_2 = gaussxwab(N_2, 0, x_0)
    N_2_result = gaussian_quad(period_integrand(x_0, mass, spring_const), N_2, 0, x_0)

    print("Boring Value: ", boring_val)
    print("8 steps integration :", N_1_result)
    print("N-1 fraction error: ", get_fraction_err(N_1_result, boring_val) * 100)

    print("-----------------------------------------------------------")
    print("Boring Value: ", boring_val)
    print("16 steps integration:", N_2_result)
    print("N-2 fraction error: ", get_fraction_err(N_2_result, boring_val) * 100)

    # ---------- Plot 1: Unweighted f(x) ----------
    f_vals_1 = [4 / spring_v_func(x, x_0, mass, spring_const) for x in x_vals_1]
    f_vals_2 = [4 / spring_v_func(x, x_0, mass, spring_const) for x in x_vals_2]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals_1, f_vals_1, "o-", label="N=8")
    plt.plot(x_vals_2, f_vals_2, "s-", label="N=16")
    plt.xlabel("x (m)")
    plt.ylabel(r"$f(x) = 4/v(x)$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---------- Plot 2: Weighted contributions ----------
    f_w_1 = [f_vals_1[i] * weights_1[i] for i in range(len(x_vals_1))]
    f_w_2 = [f_vals_2[i] * weights_2[i] for i in range(len(x_vals_2))]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals_1, f_w_1, "o-", label="N=8")
    plt.plot(x_vals_2, f_w_2, "s-", label="N=16")
    plt.xlabel("x (m)")
    plt.ylabel(r"$f4 * w_i / v(x)$")
    plt.grid(True)
    plt.legend()
    plt.show()
