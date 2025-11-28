"""
Lab 08
Question 2: FTCS solution of the wave equation
Author: Peter Burgess (November 2025)
Purpose: Solve the wave equation using FTCS method.

Outputs: animation of solution from t = 0 to t = 0.005.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
L = 1.0
v = 100.0
d = 0.10
C = 1.0
sigma = 0.03

dx = 0.002
x = np.arange(0.0, L + dx, dx)
N = len(x)

dt = 1e-6
t_final = 100e-3  # 0.005 is a good time
n_steps = int(np.ceil(t_final / dt))

# tuning (increased amplification)
frame_skip = 100  # show every n computed steps
vis_scale = 1  # vertical amplification for visibility
anim_interval_ms = 1000 // 60

# Instability detection
instability_threshold = 1  # 0.01 good

# Initial fields
psi = np.zeros(N)  # displacement
phi = C * x * (L - x) / L * np.exp(- (x - d) ** 2 / (2 * sigma ** 2))  # initial velocity
psi[0] = psi[-1] = 0.0
phi[0] = phi[-1] = 0.0


# FTCS step function
def ftcs_step(psi, phi, dt, dx, v):
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx ** 2
    psi_new = psi + dt * phi
    phi_new = phi + dt * (v ** 2) * lap
    psi_new[0] = psi_new[-1] = 0.0
    phi_new[0] = phi_new[-1] = 0.0
    return psi_new, phi_new


# Plotting
fig, ax = plt.subplots(figsize=(9, 4))
line, = ax.plot(x, vis_scale * psi, lw=2, color='C0')
ax.set_xlim(0, L)
# y-limits
ax.set_ylim(-0.0004, 0.0004)
ax.set_xlabel('x (m)')
ax.set_ylabel('psi (m)')
print('t = 0.000000 s')

# Animation state
global_step = 0
running = True

# compute number of frames
frames_to_show = max(1, n_steps // frame_skip)

def update(frame):
    global psi, phi, global_step, running
    if not running:
        return (line,)

    # advance frame_skip
    for _ in range(frame_skip):
        psi, phi = ftcs_step(psi, phi, dt, dx, v)
        global_step += 1
        if global_step >= n_steps:
            running = False
            break

    # update plot data
    line.set_ydata(vis_scale * psi)
    print(f"t = {global_step * dt:.6f} s  — showing every {frame_skip} steps")

    # instability detection
    if np.max(np.abs(psi)) * vis_scale > instability_threshold:
        print("[Stopped: instability threshold reached]")
        running = False

    return line,


anim = FuncAnimation(fig, update, frames=frames_to_show, interval=anim_interval_ms, blit=True)
plt.show()

# ============================================================
# SNAPSHOTS AT t = 2, 4, 6, 12, 100 ms
# ============================================================

snapshot_times = np.array([2, 4, 6, 12, 100]) * 1e-3   # seconds
snapshot_solutions = {T: None for T in snapshot_times}

# Reset fields for snapshot simulation
psi_s = np.zeros(N)
phi_s = C * x * (L - x) / L * np.exp(-(x - d)**2 / (2 * sigma**2))
psi_s[0] = psi_s[-1] = 0.0
phi_s[0] = phi_s[-1] = 0.0

current_time = 0.0
steps_total = int(0.100 / dt)   # simulate up to 0.1 seconds (100 ms)

print("\nGenerating snapshots...\n")

for step in range(steps_total):

    # Advance solver
    psi_s, phi_s = ftcs_step(psi_s, phi_s, dt, dx, v)
    current_time += dt

    # Save snapshots when times match
    for T in snapshot_times:
        if snapshot_solutions[T] is None and abs(current_time - T) < dt/2:
            snapshot_solutions[T] = psi_s.copy()   # Save displacement
            print(f"Saved snapshot at t = {T*1000:.1f} ms")

# ---------------------------------------------------------
# Plot and save snapshots
# ---------------------------------------------------------
for T in snapshot_times:

    y_mm = snapshot_solutions[T] * 1000.0   # convert to mm
    raw_ymax = np.max(np.abs(y_mm))
    if raw_ymax == 0:
        raw_ymax = 1e-6

    # ---- Extract exponent for scientific notation ----
    exponent = int(np.floor(np.log10(raw_ymax)))
    scale_factor = 10 ** exponent

    # ---- Scale y data to readable range ----
    y_scaled = y_mm / scale_factor
    ymax_scaled = np.max(np.abs(y_scaled))

    # ---- Nice symmetric ticks ----
    yticks = np.linspace(-ymax_scaled, ymax_scaled, 5)

    plt.figure(figsize=(8,4))
    plt.plot(x, y_scaled, lw=2)

    plt.title(f"FTCS Displacement at t = {T*1000:.1f} ms", pad=25)
    plt.xlabel("x (m)")
    plt.ylabel(f'ψ(x,t) (scaled) (mm) ×10^{exponent}')

    plt.ylim(-ymax_scaled, ymax_scaled)
    plt.yticks(yticks, [f"{val:.3f}" for val in yticks])

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"snapshot_{int(T*1000)}ms.png", dpi=300)
    plt.show()

print("\nAll snapshots saved.\n")
