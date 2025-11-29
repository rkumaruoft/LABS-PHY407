import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(8675309)

def plain_estimate(N):
    # Uniform on [0,10]
    x = np.random.rand(N) * 10.0
    f = np.exp(-2.0 * np.abs(x - 5.0))
    return 10.0 * np.mean(f)   # factor 10 for interval length

def is_estimate(N):
    # Importance sampling from N(5,1)
    x = np.random.normal(loc=5.0, scale=1.0, size=N)
    # f(x) on [0,10], zero outside
    f = np.exp(-2.0 * np.abs(x - 5.0)) * ((x >= 0.0) & (x <= 10.0))
    p = norm.pdf(x, loc=5.0, scale=1.0)
    # avoid division by zero
    return np.mean(f / p)

# Parameters
N = 10000
repeats = 100

# Run experiments
I_plain = np.array([plain_estimate(N) for _ in range(repeats)])
I_is    = np.array([is_estimate(N) for _ in range(repeats)])

# Print summary statistics
exact = 1.0 - np.exp(-10.0)
print("Exact value: {:.8f}".format(exact))
print("Mean-value method: mean = {:.8f}, std = {:.8f}".format(I_plain.mean(), I_plain.std(ddof=1)))
print("Importance sampling: mean = {:.8f}, std = {:.8f}".format(I_is.mean(), I_is.std(ddof=1)))

# Histogram range chosen to centre around the exact value
hist_range = [0.995, 1.005]

# First figure: mean-value method
plt.figure(figsize=(6,4))
plt.hist(I_plain, bins=10, range=hist_range, color='C0', edgecolor='black')
plt.axvline(exact, color='k', linestyle='--', linewidth=1)
plt.title('Mean-value method (Uniform on [0,10])')
plt.xlabel('Integral estimate')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

# Second figure: importance sampling
plt.figure(figsize=(6,4))
plt.hist(I_is, bins=10, range=hist_range, color='C1', edgecolor='black')
plt.axvline(exact, color='k', linestyle='--', linewidth=1)
plt.title('Importance sampling (Normal(5,1))')
plt.xlabel('Integral estimate')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()
