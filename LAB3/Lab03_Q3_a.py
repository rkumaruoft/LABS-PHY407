import os
from common_funcs import *


def exp_2(x):
    return np.exp(-x ** 2)


def derivative_exp_2(x):
    return -2 * x * np.exp(-x ** 2)


def forward_derivative(function_f, point_x, step_h):
    return (function_f(point_x + step_h) - function_f(point_x)) / step_h


def central_derivative(function_f, point_x, step_h):
    return (function_f(point_x + (step_h / 2)) - function_f(point_x - (step_h / 2))) / step_h


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    # 10^-16 up to 10^0
    h_range = [10.0 ** i for i in range(-16, 1)]

    point_x = 0.5
    analytic_derivative = derivative_exp_2(0.5)

    print(f"Analytic Derivative: {analytic_derivative}")

    forwards = []
    forward_errors = []
    centrals = []
    central_errors = []
    for h in h_range:
        this_forward = forward_derivative(exp_2, 0.5, h)
        this_forward_error = abs(this_forward - analytic_derivative)
        this_central = central_derivative(exp_2, 0.5, h)
        this_central_error = abs(this_central - analytic_derivative)
        print(f"-----------------------h = {h}---------------------")
        print(f"forward derivative = {this_forward} and error = {this_forward_error}")
        print(f"central derivative = {this_central} and error = {this_central_error}")
        forwards.append(this_forward)
        forward_errors.append(this_forward_error)
        centrals.append(this_central)
        central_errors.append(this_central_error)

    plt.figure()
    plt.loglog(h_range, forward_errors, marker=".", linestyle="-", label="Forward difference error")
    plt.loglog(h_range, central_errors, marker=".", linestyle="--", label="Central difference error")
    plt.axvline(x=10**-8)
    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/Errors.png", dpi=300, bbox_inches="tight")
    plt.show()
