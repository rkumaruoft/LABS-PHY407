import numpy as np
import time

# number of bins
M = 1000
bin_min = -5
bin_max = 5
bin_width = (bin_max - bin_min) / M

Ns = [10, 100, 1000, 10000, 100000, 1000000]
times = []

for N in Ns:
    # edges for bins
    edges = np.zeros(M + 1)
    for i in range(M + 1):
        edges[i] = bin_min + (i * bin_width)
    num_edges = edges.shape[0]

    samples = np.random.randn(N)
    counts = np.zeros(M)

    start = time.time()
    for sample in samples:
        if sample == bin_max:
            counts[-1] += 1
        else:
            for i in range(num_edges - 1):
                if edges[i] <= sample < edges[i + 1]:
                    counts[i] += 1
                    break
    end = time.time()
    times.append(end - start)
print(times)
for i in range(len(Ns)):
    print("Time taken for " + str(Ns[i]) + " samples:" + str(times[i]))
