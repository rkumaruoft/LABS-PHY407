
import numpy as np
import matplotlib.pyplot as plt

array = np.loadtxt('cdata.txt')

true_std = np.std(array, ddof=1)

def method_1_std(array1):
    sum1 = 0
    for item in array1:
        sum1 += item
    array1_mean = sum1/len(array1)

    print(array1_mean)
    print(np.mean(array))

    var1 = 0
    for item in array1:
        var1 = (item - array1_mean)**2
    return np.sqrt((var1/(len(array1)-1)))


print(method_1_std(array))

#print(true_std)
#print(array)