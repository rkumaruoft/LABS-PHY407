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
t_final = 0.005 # 0.005 is a good time
n_steps = int(np.ceil(t_final / dt))

# tuning (increased amplification)
frame_skip = 100       # show every n computed steps
vis_scale = 1     # vertical amplification for visibility
anim_interval_ms = 1000 // 60

# Instability detection
instability_threshold = 1 # 0.01 good

# Initial fields
psi = np.zeros(N)     # displacement
phi = C * (L - x) / L * np.exp(- (x - d)**2 / (2 * sigma**2))  # initial velocity
psi[0] = psi[-1] = 0.0
phi[0] = phi[-1] = 0.0

# FTPS step function
def ftcs_step(psi, phi, dt, dx, v):
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx**2
    psi_new = psi + dt * phi
    phi_new = phi + dt * (v**2) * lap
    psi_new[0] = psi_new[-1] = 0.0
    phi_new[0] = phi_new[-1] = 0.0
    return psi_new, phi_new


# Plotting
fig, ax = plt.subplots(figsize=(9, 4))
line, = ax.plot(x, vis_scale * psi, lw=2, color='C0')
ax.set_xlim(0, L)
# y-limits
ax.set_ylim(-0.01, 0.01)
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

    return (line,)

anim = FuncAnimation(fig, update, frames=frames_to_show, interval=anim_interval_ms, blit=True)
plt.show()

# ============================================================
# Generate FTCS snapshots at the same times as the spectral method
# ============================================================

print("\nGenerating FTCS snapshots...\n")

snapshot_times = np.array([2, 4, 6, 12, 100]) * 1e-3   # seconds
snapshot_solutions = {t: None for t in snapshot_times}

# FTCS parameters (same as before)
psi_ftcs = np.zeros(N)
phi_ftcs = C * (L - x) / L * np.exp(-(x - d)**2 / (2*sigma**2))
psi_ftcs[0] = psi_ftcs[-1] = 0
phi_ftcs[0] = phi_ftcs[-1] = 0

def ftcs_step_only(psi, phi):
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    psi_new = psi + dt * phi
    phi_new = phi + dt * v**2 * lap
    psi_new[0] = psi_new[-1] = 0
    phi_new[0] = phi_new[-1] = 0
    return psi_new, phi_new

# Run FTCS long enough to hit 100 ms
steps_total = int(0.100 / dt)
time_ftcs = 0.0

for step in range(steps_total):
    psi_ftcs, phi_ftcs = ftcs_step_only(psi_ftcs, phi_ftcs)
    time_ftcs += dt

    # store snapshots
    for target in snapshot_times:
        if snapshot_solutions[target] is None and abs(time_ftcs - target) < dt/2:
            snapshot_solutions[target] = psi_ftcs.copy()
            print(f"Saved snapshot at t = {target*1000:.1f} ms")

# ============================================================
# Plot each snapshot
# ============================================================

for t in snapshot_times:
    plt.figure(figsize=(8,4))
    plt.plot(x, snapshot_solutions[t], lw=2)
    plt.title(f"FTCS Solution at t = {t*1000:.1f} ms")
    plt.xlabel("x (m)")
    plt.ylabel("ψ(x,t) (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ftcs_{int(t*1000)}ms.png", dpi=300)
    plt.show()

print("\nAll FTCS snapshots saved.\n")
