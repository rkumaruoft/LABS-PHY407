"""
Importance sampling
Purpose: To evaluate given rapidly varying integrands using importance sampling.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8675309)

def plain_estimate(N):
    x = np.random.rand(N)
    return np.mean(x**(-0.5) / (1 + np.exp(x)))

def is_estimate(N):
    u = np.random.rand(N)
    x = u**2
    return np.mean(2.0 / (1 + np.exp(x)))

N = 10000
repeats = 100
I_plain = np.array([plain_estimate(N) for _ in range(repeats)])
I_is    = np.array([is_estimate(N) for _ in range(repeats)])

# First figure: mean-value method
plt.figure(figsize=(6,4))
plt.hist(I_plain, 10, range=[0.8,0.88], color='C0', edgecolor='black')
plt.title('Mean-value method')
plt.xlabel('Integral estimate')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

# Second figure: importance sampling
plt.figure(figsize=(6,4))
plt.hist(I_is, 10, range=[0.8,0.88], color='C1', edgecolor='black')
plt.title('Importance sampling')
plt.xlabel('Integral estimate')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

