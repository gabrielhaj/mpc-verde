import pandas as pd
import matplotlib.pyplot as plt
import mpctools.plots as mpcplots
import numpy as np

m = pd.read_excel("1exemplo.xlsx")
p = pd.read_excel("2exemplo.xlsx")
q = pd.read_excel("3exemplo.xlsx")
x1 = m["x"]
y1 = m["y"]
theta1 = m["theta"]
v1 = m["v"]
w1 = m["w"]
t = m["t"]
x2 = p["x"]
y2 = p["y"]
theta2 = p["theta"]
v2 = p["v"]
w2 = p["w"]
x3 = q["x"]
y3 = q["y"]
theta3 = q["theta"]
v3 = q["v"]
w3 = q["w"]
t3 = q["t"]

fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(t,x1,label='Multiple-shooting')
axs[0, 0].plot(t,x2,label='Single-shooting')
#axs[0, 0].plot(t3,x3,label='MPCTools')
axs[0,0].legend()
axs[0, 0].set_ylabel("x[m]")
axs[0, 0].set_xlabel("t[s]")
axs[1, 0].plot(t,y1)
axs[1, 0].plot(t,y2)
#axs[1, 0].plot(t3,y3)
axs[1, 0].set_ylabel("y[m]")
axs[1, 0].set_xlabel("t[s]")
axs[0, 1].step(t,v1)
axs[0, 1].step(t,v2)
#axs[0, 1].step(t3,v3)
axs[0, 1].set_ylabel("v[m/s]")
axs[0, 1].set_xlabel("t[s]")
axs[2, 0].plot(t,theta1)
axs[2, 0].plot(t,theta2)
#axs[2, 0].plot(t3,theta3)
axs[2, 0].set_ylabel("theta[rad]")
axs[2, 0].set_xlabel("t[s]")
axs[1,1].step(t,w1)
axs[1,1].step(t,w2)
#axs[1,1].step(t3,w3)
axs[1, 1].set_ylabel("w[rad/s]")
axs[1, 1].set_xlabel("t[s]")
axs[2,1].set_visible(False)
axs[3,0].set_visible(False)
axs[3,1].set_visible(False)


plt.show()
mpcplots.showandsave(fig, "Single-Multiple_comparison.pdf")
