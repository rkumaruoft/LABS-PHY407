""" scipy.io.wavfile allows you to read and write .wav files """
from scipy.io.wavfile import read, write
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt

CUTOFF_HZ = 880

# read the data into two stereo channels
# sample is the sampling rate, data is the data in each channel,
# dimensions [2, nsamples]
sample, data = read('GraviteaTime.wav')

# sample is the sampling frequency, 44100 Hz
# separate into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
N_Points = len(channel_0)

#time axis in seconds
nsamples = data.shape[0]
time_index = np.arange(nsamples) / float(sample)

#convert integer audio to float in [-1, 1] for plotting amplitude
if np.issubdtype(data.dtype, np.integer):
    dtype_info = np.iinfo(data.dtype)
    scale = max(abs(dtype_info.min), abs(dtype_info.max))
    channel_0_plot = channel_0.astype(np.float32) / scale
    channel_1_plot = channel_1.astype(np.float32) / scale
else:
    channel_0_plot = channel_0.astype(np.float32)
    channel_1_plot = channel_1.astype(np.float32)

#Plot channel 0
plt.figure(figsize=(10, 3.5))
plt.plot(time_index, channel_0_plot, linewidth=0.6, color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(time_index[0], time_index[-1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Plot channel 1
plt.figure(figsize=(10, 3.5))
plt.plot(time_index, channel_1_plot, linewidth=0.6, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(time_index[0], time_index[-1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

channel_0_plot = channel_0.copy()
channel_1_plot = channel_1.copy()

#FFT of each channel
fft0 = np.fft.rfft(channel_0)
fft1 = np.fft.rfft(channel_1)
freqs = np.fft.rfftfreq(nsamples, d=1.0 / sample)

#original FFT amplitudes
fft0_amp_original = np.abs(fft0)
fft1_amp_original = np.abs(fft1)

#build mask and apply
mask = freqs <= CUTOFF_HZ
fft0_filtered = fft0 * mask
fft1_filtered = fft1 * mask

fft0_amp_filt = np.abs(fft0_filtered)
fft1_amp_filt = np.abs(fft1_filtered)

#inverse FFT to get filtered
channel_0_filt = np.fft.irfft(fft0_filtered, n=nsamples)
channel_1_filt = np.fft.irfft(fft1_filtered, n=nsamples)

channel_0_filt_plot = channel_0_filt.copy()
channel_1_filt_plot = channel_1_filt.copy()

#write filtered output
data_out = empty(data.shape, dtype=data.dtype)
data_out[:, 0] = channel_0_filt
data_out[:, 1] = channel_1_filt

write("GraviteaTime_lpf", sample, data_out)

