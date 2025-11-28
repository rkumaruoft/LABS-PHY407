from dcst import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    times = np.array([2.0, 4.0, 6.0, 12.0, 100.0]) * 1e-3  # in seconds

    # define constants
    N = 1000  # interior points
    L = 1.0  # meter

    x_arr = np.linspace(0, L, N + 1)

    # initial position array
    phi_0 = np.zeros(N + 1)

    # Hammer strike constants
    d = 0.1  # meter
    C = 1  # m/s
    sigma = 0.3  # m
    v = 100  # m/s

    # initial velocity array
    psi_0 = C * (x_arr * (L - x_arr) / (L ** 2)) * np.exp(-((x_arr - d) ** 2) / (2 * (sigma ** 2)))

    # compute coefficients of phi_0 and psi_0
    phi0_k = dst(phi_0)
    psi0_k = dst(psi_0)

    # compute omega array
    k = np.arange(N + 1)
    omega_arr = v * k * np.pi / L

    phi_x_t = []
    for t in times:
        Phi_modes = phi0_k[1:] * np.cos(omega_arr[1:] * t) + (psi0_k[1:] / omega_arr[1:]) * np.sin(omega_arr[1:] * t)
        # full coefficient array
        Phi_k_t = np.array([0] + list(Phi_modes))
        # reconstruct φ(x,t)
        phi_x_t.append(idst(Phi_k_t))

    plt.figure(figsize=(8, 4))

    for i, t in enumerate(times):
        # plot curve and capture the line object
        line, = plt.plot(x_arr, phi_x_t[i], linewidth=1, label=f't = {t}')

        # get the curve color
        c = line.get_color()

    plt.xlabel("x (m)")
    plt.ylabel("φ(x,t) (mm)")

    # global min/max across all curves
    ymin = min(np.min(arr) for arr in phi_x_t)
    ymax = max(np.max(arr) for arr in phi_x_t)

    # include zero in the range
    ymin = min(ymin, 0)
    ymax = max(ymax, 0)

    # make symmetric range for nice spacing
    ymax_abs = max(abs(ymin), abs(ymax))
    ymin = -ymax_abs
    ymax = ymax_abs

    # choose even number of ticks (e.g., 5 or 7)
    num_ticks = 7
    yticks_m = np.linspace(ymin, ymax, num_ticks)

    # convert labels to mm
    yticks_mm = yticks_m * 1000
    plt.yticks(yticks_m, [f"{y:.3f}" for y in yticks_mm])

    plt.title("String displacement at various times")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("wave.png", dpi=300)
    plt.show()

    # ANIMATION (Un-comment the block below to show animation)
    # # new time array for animation
    # t_anim = np.arange(0, 0.100 + 0.002, 0.001)   # 0 → 100 ms in 2 ms steps
    #
    # phi_anim = []
    # for t in t_anim:
    #     Phi_modes = (
    #         phi0_k[1:] * np.cos(omega_arr[1:] * t)
    #         + (psi0_k[1:] / omega_arr[1:]) * np.sin(omega_arr[1:] * t)
    #     )
    #     Phi_k_t = np.array([0] + list(Phi_modes))
    #     phi_anim.append(idst(Phi_k_t))
    #
    # fig, ax = plt.subplots(figsize=(8, 4))
    # line, = ax.plot(x_arr, phi_anim[0])
    # ax.set_ylim(-0.0004, 0.0004)
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("φ(x,t)")
    # ax.set_title("String Vibration Animation (t = 0 ms)")
    #
    # def update(frame):
    #     line.set_ydata(phi_anim[frame])
    #     ax.set_title(f"t = {t_anim[frame] * 1000:.0f} ms")
    #     return line,
    #
    # ani = FuncAnimation(fig, update, frames=len(t_anim), interval=60)
    # plt.show()

    # save snapshots seperately
    for i, t in enumerate(times):

        # convert to mm
        y = (phi_x_t[i] * 1000).copy()  # meters → mm

        # Force zero line if amplitude is negligible (e.g., 100 ms)
        if np.max(np.abs(y)) < 1e-6:
            y = np.zeros_like(y)

        plt.figure(figsize=(8, 4))
        plt.plot(x_arr, y, linewidth=2)

        plt.xlabel("x (m)")
        plt.ylabel("φ(x,t) (mm)")

        # Set symmetric y-axis around zero with evenly spaced ticks
        ymax = np.max(np.abs(y))
        yticks = np.linspace(-ymax, ymax, 5)
        plt.yticks(yticks, [f"{val:.3f}" for val in yticks])

        plt.title(f"Spectral Method Snapshot at t = {t * 1000:.1f} ms")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"spectral_{int(t * 1000)}ms.png", dpi=300)
        plt.close()
