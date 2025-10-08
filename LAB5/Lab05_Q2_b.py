""" scipy.io.wavfile allows you to read and write .wav files """
from scipy.io.wavfile import read, write
from numpy import empty
# read the data into two stereo channels
# sample is the sampling rate, data is the data in each channel,
# dimensions [2, nsamples]
sample, data = read('input_file.wav')
# sample is the sampling frequency, 44100 Hz
# separate into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
N_Points = len(channel_0)
# ... do work on the data...
# this creates an empty array data_out with the same shape as "data"
# (2*N_Points) and the same type as "data" (int16)
data_out = empty(data.shape, dtype = data.dtype)
# fill data_out
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out
write('output_file.wav', sample, data_filt)