from cmath import sqrt
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
g = pd.read_csv("lane_change.csv")
a = g["x"]
b = g["y"]
c = g["uref"]

#trecho 2
v = 0.6
k = 500
dt = 0.05
w = np.pi/(k*dt)
r = v/w
t = np.linspace((3/2)*np.pi,(5/2)*np.pi,k)
x_2 = a[499] + r*np.cos(t)
y_2 = b[499] + r + r*np.sin(t)
kt = 500 + int(k)
#trecho 3
ds = 10
k = ds/(v*dt)
t = np.linspace(0,ds,k)
x_3 = x_2[-1] - t*v
y_3 = y_2[-1] + np.zeros(int(k))
kt = kt + int(k)
#trecho4
w = v/(r/2)
k = np.pi/(w*dt)
t = np.linspace(np.pi/2,(3/2)*np.pi,k)
x_4 = x_3[-1] + (r/2)*np.cos(t)
y_4 = y_3[-1] - r/2 + (r/2)*np.sin(t)
kt = kt + int(k)
#trecho5
t = np.linspace(np.pi/2,-np.pi/2,k)
x_5 = x_4[-1] + (r/2)*np.cos(t)
y_5 = y_4[-1] - (1/2)*r + (r/2)*np.sin(t)
kt = kt+ int(k)
#trecho6
d = x_5[-1]
k = d/(v*dt)
t = np.linspace(0,k*dt,k)
x_6 = d - v*t
y_6 = y_5[-1] + np.zeros(int(k))
kt = kt + int(k)
#trecho7
r = y_6[-1]/2
w = v/r
k = np.pi/(w*dt)
t = np.linspace(np.pi/2,(3/2)*np.pi,k)
x_7 = x_6[-1] + r*np.cos(t)
y_7 = y_6[-1] - r + r*np.sin(t)
kt = kt + int(k)

#total
x_t = np.hstack((a,x_2[1:],x_3[1:],x_4[1:],x_5[1:],x_6[1:],x_7[1:]))
y_t = np.hstack((b,y_2[1:],y_3[1:],y_4[1:],y_5[1:],y_6[1:],y_7[1:]))
z = np.zeros(x_t.size)
phi = np.zeros(x_t.size)
for n in range(x_t.size):
    if(n == 0):
        phi[n] = 0
    else:
        if(np.arctan2(y_t[n]-y_t[n-1],x_t[n]-x_t[n-1]) < 0):
            phi[n] = np.arctan2(y_t[n]-y_t[n-1],x_t[n]-x_t[n-1]) + 2*np.pi
        else:
            phi[n] = np.arctan2(y_t[n]-y_t[n-1],x_t[n]-x_t[n-1])    
#plt.plot(phi)
print(kt)
print(x_t.size)
plt.plot(x_t,y_t)
plt.show()
c2 = np.zeros(x_t.size)
c2[0:500] = c
c2[500:] = v
h = np.vstack((x_t,y_t,c2))
h = h.T
g2 = pd.DataFrame(h)
g2.to_csv('out.csv',index = False)


