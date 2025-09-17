import numpy as np
import matplotlib.pyplot as plt

array = np.loadtxt('cdata.txt')

true_std = np.std(array, ddof=1)


def method_1_std(array1):
    sum1 = 0
    for item in array1:
        sum1 += item
    array1_mean = sum1 / len(array1)

    var1 = 0
    for item in array1:
        var1 += (item - array1_mean) ** 2
    return np.sqrt((var1 / (len(array1) - 1)))


def method_2_std(array):
    n = len(array)
    sum_x = 0
    sum_x_squared = 0

    for item in array:
        sum_x += item
        sum_x_squared += (item ** 2)

    numerator = sum_x_squared - ((sum_x ** 2) / n)
    return np.sqrt(numerator / (n - 1))

def get_relative_error(value1, value2):
    return abs(value1 - value2)/value2


print("method 1 std: ", method_1_std(array))
print("method 2 std: ", method_2_std(array))
print("true std: ", true_std)

print("relative error method 1:", get_relative_error(method_1_std(array), true_std))
print("relative error method 2:", get_relative_error(method_2_std(array), true_std))

#print(array)
