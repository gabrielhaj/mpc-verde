import pandas as pd
import matplotlib.pyplot as plt
import mpctools.plots as mpcplots
import numpy as np

m = pd.read_excel("Pend_data.xlsx",na_values = ["NA"])
p = pd.read_excel("invertpend_data_py.xlsx")
x = p["x"]
x_dot =p["x_dot"]
theta = p["theta"]
theta_dot = p["theta_dot"]
u = p["u"]
t = p["t"]
mx = m["x"][1001:12150:11]
mx = mx[:-13]
mxdot = m["x_dot"][1001:12150:11]
mxdot = mxdot[:-13]
mtheta = m["theta"][1001:12150:11]
mtheta = mtheta[:-13]
mthetadot = m["theta_dot"][1001:12150:11]
mthetadot = mthetadot[:-13]
um = m["F"][100:1001]
um = np.append(um, np.zeros(100))
fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(t,x)
#axs[0, 0].plot(t,mx)
axs[0, 0].set_ylabel("x[m]")
axs[0, 0].set_xlabel("t[s]")
axs[1, 0].plot(t,x_dot)
#axs[1, 0].plot(t,mxdot)
axs[1, 0].set_ylabel("v[m/s]")
axs[1, 0].set_xlabel("t[s]")
axs[0, 1].step(t,u)
#axs[0, 1].plot(t,um)
axs[0, 1].set_ylabel("F[N]")
axs[0, 1].set_xlabel("t[s]")
axs[2, 0].plot(t,theta)
#axs[2, 0].plot(t,mtheta)
axs[2, 0].set_ylabel("theta[rad]")
axs[2, 0].set_xlabel("t[s]")
axs[3, 0].plot(t,theta_dot)
#axs[3, 0].plot(t,mthetadot,label = "matlab")
axs[3, 0].set_ylabel("w[rad/s]")
axs[3, 0].set_xlabel("t[s]")
axs[3, 0].legend()
axs[1,1].set_visible(False)
axs[2,1].set_visible(False)
axs[3,1].set_visible(False)
plt.show()
mpcplots.showandsave(fig, "ex.png")