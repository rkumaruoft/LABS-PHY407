import numpy as np

# Simple relaxation method to solve x = 1 - exp(-c * x)
def relaxation(c, initial_guess, convergence=1e-6, max_iter=10000):
    x = initial_guess   # start from initial guess
    iterations = 0
    while iterations < max_iter:
        # Iteration formula for relaxation method
        x_new = 1 - np.exp(-c * x)
        iterations += 1

        # Check convergence condition
        if abs(x_new - x) <= convergence:
            return x_new, iterations  # return solution and number of steps

        # Update x for next iteration
        x = x_new

    # If loop finishes without convergence
    return -1, -1


# Over-relaxation method (adds acceleration factor omega)
def overrelaxation(c, initial_guess, omega, convergence=1e-6, max_iter=10000):
    x = initial_guess   # start from initial guess
    iterations = 0
    while iterations < max_iter:
        # Over-relaxation iteration formula
        x_new = ((1 + omega) * (1 - np.exp(-c * x))) - (omega * x)
        iterations += 1

        # Check convergence condition
        if abs(x_new - x) <= convergence:
            return x_new, iterations  # return solution and number of steps

        # Update x for next iteration
        x = x_new

        # Print progress for debugging
        print(f"Current value {x} in {iterations} iterations.")

    # If loop finishes without convergence
    return -1, -1


if __name__ == "__main__":
    # Run simple relaxation method
    value, iterations = relaxation(c=2, initial_guess=1)
    print("-------------------Simple Relaxation---------------------")
    print(f"Value converges to {value} in {iterations} iterations.\n")

    omega = 0.7
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")

    omega = 0.8
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")

    omega = 1
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")

    omega = 0.1
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")

    # Run over-relaxation method with omega = 0.5
    omega = 0.3
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")

    # Run over-relaxation method with omega = 0.5
    omega = 0.5
    print(f"-------------------Over Relaxation (omega = {omega})---------------------")
    value, iterations = overrelaxation(c=2, initial_guess=1.0, omega=omega)
    print(f"Value converges to {value} in {iterations} iterations.")
