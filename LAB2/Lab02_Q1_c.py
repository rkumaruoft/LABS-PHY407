import numpy as np
from Lab02_Q1 import method_1_std

if __name__ == "__main__":
    mean_1, sigma_1, n_1 = 0., 1., 2000
    mean_2, sigma_2, n_2 = 0., 1., 2000

    sequence_1 = np.random.normal(mean_1, sigma_1, n_1)
    sequence_2 = np.random.normal(mean_2, sigma_2, n_2)

