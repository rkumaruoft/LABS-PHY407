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
