import numpy as np
import matplotlib.pyplot as plt

#Load data
SLP = np.loadtxt('SLP.txt')
lon = np.loadtxt('lon.txt')
times = np.loadtxt('times.txt')

SLP_anom = SLP - SLP.mean(axis=1, keepdims=True)

#FFT in longitude
SLP_hat = np.fft.fft(SLP_anom, axis=1)

def reconstruct_m(SLP_hat, m):
    nlon = SLP_hat.shape[1]
    m_idx = m % nlon
    hat = np.zeros_like(SLP_hat, dtype=complex)
    hat[:, m_idx] = SLP_hat[:, m_idx]
    if m_idx != 0 and not (nlon % 2 == 0 and m_idx == nlon // 2):
        hat[:, (-m_idx) % nlon] = SLP_hat[:, (-m_idx) % nlon]
    return np.fft.ifft(hat, axis=1).real

#Reconstruct fields
field_m3 = reconstruct_m(SLP_hat, 3)
field_m5 = reconstruct_m(SLP_hat, 5)

#Create grid for plotting
Lon_grid, Time_grid = np.meshgrid(lon, times)

#Figure for m = 3
fig1, ax1 = plt.subplots(figsize=(9, 4))
cf1 = ax1.contourf(Lon_grid, Time_grid, field_m3, levels=21, extend='both')
ax1.contour(Lon_grid, Time_grid, field_m3, levels=[0], colors='k', linewidths=0.6)
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Time (days since Jan 1 2015)')
print('SLP component m = 3')
cbar1 = fig1.colorbar(cf1, ax=ax1)
cbar1.set_label('hPa')
plt.tight_layout()

#Separate figure for m = 5
fig2, ax2 = plt.subplots(figsize=(9, 4))
cf2 = ax2.contourf(Lon_grid, Time_grid, field_m5, levels=21, extend='both')
ax2.contour(Lon_grid, Time_grid, field_m5, levels=[0], colors='k', linewidths=0.6)
ax2.set_xlabel('Longitude (deg)')
ax2.set_ylabel('Time (days since Jan 1 2015)')
print('SLP component m = 5')
cbar2 = fig2.colorbar(cf2, ax=ax2)
cbar2.set_label('hPa')
plt.tight_layout()

plt.show()
