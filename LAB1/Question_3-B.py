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

manual_times = np.array([0.027260541915893555, 0.014522790908813477, 0.13973212242126465,
                         1.4382565021514893, 12.63277006149292, 136.6366982460022])
numpy_times = times

plt.figure(figsize=(8, 6))
plt.plot(Ns, manual_times, marker='o', label="Manual Times")
plt.plot(Ns, numpy_times, marker='s', label="NumPy Times")

plt.yscale("log")  # log scale only on y-axis

plt.xlabel("Number of samples (N)")
plt.ylabel("Time (seconds, log scale)")
plt.legend()
# Save figure as PNG
plt.savefig("histogram_timing_comparison.png", dpi=300, bbox_inches="tight")

plt.show()

