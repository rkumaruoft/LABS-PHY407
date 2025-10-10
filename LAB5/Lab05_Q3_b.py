import numpy as np
import matplotlib.pyplot as plt

#parameters
m = 5
theta_deg = 45.0
rE = 6.371e6
sec_per_day = 86400.0

SLP = np.loadtxt('SLP.txt')
lon = np.loadtxt('lon.txt')
times = np.loadtxt('times.txt')

ntime, nlon = SLP.shape
SLP_anom = SLP - SLP.mean(axis=1, keepdims=True)

#Fourier transform longitude
SLP_hat = np.fft.fft(SLP_anom, axis=1)

m_index = m % nlon

#complex amplitude for each time
A_m_t = SLP_hat[:, m_index]

#compute instantaneous phase
phi = np.angle(A_m_t)
phi_unwrap = np.unwrap(phi)

#Compute dphi/dt (radians per day)
dphi_dt = np.gradient(phi_unwrap, times)

#phase speed
dlambda_dt_rad_per_day = - (1.0 / m) * dphi_dt

#convertions
dlambda_dt_deg_per_day = np.degrees(dlambda_dt_rad_per_day)
circumference_at_theta = 2.0 * np.pi * rE * np.cos(np.radians(theta_deg))
speed_m_per_s = (dlambda_dt_deg_per_day / 360.0) * circumference_at_theta / sec_per_day

#fit a linear slope
coeffs = np.polyfit(times, phi_unwrap, 1)
mean_dphi_dt = coeffs[0]
mean_dlambda_dt_rad_per_day = -mean_dphi_dt / m
mean_deg_per_day = np.degrees(mean_dlambda_dt_rad_per_day)
mean_speed_m_s = (mean_deg_per_day / 360.0) * circumference_at_theta / sec_per_day

print(f"Assumed latitude: {theta_deg} deg")
print(f"Mean phase speed (deg/day) for m={m}: {mean_deg_per_day:.3f} deg/day")
print(f"Mean phase speed (m/s) for m={m}: {mean_speed_m_s:.3f} m/s")
print(f"Sign convention: positive means eastward propagation")

#Plotting:
print('\n')

plt.figure()
plt.plot(times, phi_unwrap, '-k', label='unwrapped phase (rad)')
plt.plot(times, np.polyval(coeffs, times), '--r', label='linear fit')
plt.xlabel('time (days)')
plt.ylabel('phase (radians)')
print(f'Phase of Fourier coefficient m={m} (unwrapped)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(times, dlambda_dt_deg_per_day, '-b', label='instantaneous deg/day')
plt.hlines(mean_deg_per_day, times.min(), times.max(), colors='r', linestyles='--', label='mean deg/day')
plt.xlabel('time (days)')
plt.ylabel('zonal phase speed (deg/day)')
print(f'Instantaneous phase speed for m={m}')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(times, speed_m_per_s, '-g', label='instantaneous m/s')
plt.hlines(mean_speed_m_s, times.min(), times.max(), colors='r', linestyles='--', label='mean m/s')
plt.xlabel('time (days)')
plt.ylabel('phase speed (m/s)')
print(f'Instantaneous phase speed for m={m} (converted to m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
