import numpy as np
import time
import matplotlib.pyplot as plt

# number of bins
M = 1000
bin_min = -5
bin_max = 5

Ns = [10, 100, 1000, 10000, 100000, 1000000]
times = []

for N in Ns:
    samples = np.random.randn(N)

    start = time.time()
    counts, edges = np.histogram(samples, bins=M, range=(bin_min, bin_max))
    end = time.time()

    times.append(end - start)

# print timing results
for i in range(len(Ns)):
    print(f"Time taken for {Ns[i]} samples: {times[i]:.6f} seconds")

manual_times = np.array([0.001987457275390625, 0.014005661010742188, 0.17135143280029297,
                            1.340855360031128, 13.632678270339966, 204.59526944160461])
numpy_times = times

plt.figure(figsize=(8, 6))
plt.plot(Ns, manual_times, marker='o', label="Manual histogram")
plt.plot(Ns, numpy_times, marker='s', label="NumPy histogram")

plt.yscale("log")   # log scale only on y-axis

plt.xlabel("Number of samples (N)")
plt.ylabel("Time (seconds, log scale)")
plt.title("Timing Comparison: Manual vs NumPy Histogram")
plt.legend()

plt.show()

