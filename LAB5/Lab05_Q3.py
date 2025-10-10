from numpy import loadtxt
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar

SLP = loadtxt('SLP.txt')
Longitude = loadtxt('lon.txt')
Times = loadtxt('times.txt')

contourf(Longitude, Times, SLP)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()

