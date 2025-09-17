
import numpy as np
import matplotlib.pyplot as plt

array = np.loadtxt('cdata.txt')

true_std = np.std(array, ddof=1)

print(true_std)
print(array)