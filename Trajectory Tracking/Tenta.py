from turtle import end_fill


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

g = pd.read_csv("lane_change.csv")
a = g["x"]
b = g["y"]
c = g["uref"]
dt = 0.05
vel = 0.8
x = np.zeros(50)
y = np.zeros(50);
for n in range(50):
    if(n == 0):
        x[n] = 0
        y[n] = 0
    else:
        x[n] = x[n-1] + vel*dt
        y[n] = 0



t = np.linspace((3/2)*np.pi,2*np.pi,50)
x_1 = x[-1] + 1*np.cos(t)
y_1 = y[-1]+1 + 1*np.sin(t)


t = np.linspace(np.pi,(3/2)*np.pi,50)
x_2 = x_1[-1] +1 + 1*np.cos(-t)
y_2 = y_1[-1] + 1*np.sin(-t)

x_3 = np.zeros(50)
y_3 = np.zeros(50);
for n in range(50):
    if(n == 0):
        x_3[n] = x_2[-1]
        y_3[n] = y_2[-1]
    else:
        x_3[n] = x_3[n-1] + vel*dt
        y_3[n] = y_2[-1]


plt.plot(np.hstack((x[:-1],x_1[:-1],x_2[:-1],x_3)),np.hstack((y[:-1],y_1[:-1],y_2[:-1],y_3)))
plt.show()



