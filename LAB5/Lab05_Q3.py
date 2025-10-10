from numpy import loadtxt
import matplotlib.pyplot as plt

SLP = loadtxt('SLP.txt')
Longitude = loadtxt('lon.txt')
Times = loadtxt('times.txt')

plt.contourf(Longitude, Times, SLP)
plt.xlabel('longitude(degrees)')
plt.ylabel('days since Jan. 1 2015')
plt.title('SLP anomaly (hPa)')
plt.colorbar()
plt.show()


