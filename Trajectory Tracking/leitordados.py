from cmath import sqrt
from logging.handlers import NTEventLogHandler
from statistics import median_grouped
from time import time
import mpctools as mpc
import matplotlib.pyplot as plt
from mpctools.tools import DiscreteSimulator
import numpy as np
from sympy import sqrt_mod
from simulation_code import simulate
import casadi as ca
import pandas as pd

g = pd.read_csv("dados.csv")
x1 = g["x1"]
x2 = g["x2"]
x3 = g["x3"]
u = g["u"]
x = g["x"]
y = g["y"]

h = pd.read_csv("dados2.csv")
x1i = h["x1"]
x2i= h["x2"]
x3i = h["x3"]
ui = h["u"]
xi = h["x"]
yi = h["y"]
y_ref = h["yref"]
phi_ref = h["phiref"]
r_ref = h["rref"]
d_ref = h["deltaref"]

j = pd.read_csv("lane_change.csv")
a = j["x"]
b = j["y"]
c = j["uref"]

times = 0.05*500*np.linspace(0,1,500)
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(times,y_ref, label ='referencia',linestyle = 'dashed')
axs[0, 0].plot(times,x1, label ='LTV')
axs[0, 0].plot(times,x1i, label ='LTI')

axs[0, 0].set_ylabel("Posição lateral[m]")
axs[0, 0].set_xlabel("t[s]")
axs[0, 0].legend()
axs[1, 0].plot(times,phi_ref,linestyle = 'dashed')
axs[1, 0].plot(times,x2)
axs[1, 0].plot(times,x2i)

axs[1, 0].set_ylabel("Ângulo de guinada[rad]")
axs[1, 0].set_xlabel("t[s]")
axs[1, 0].legend()
axs[2, 0].plot(times,r_ref,linestyle = 'dashed')
axs[2, 0].plot(times,x3)
axs[2, 0].plot(times,x3i)

axs[2, 0].set_ylabel("Velocidade angular[rad/s]")
axs[2, 0].set_xlabel("t[s]")
axs[2, 0].legend()
axs[0, 1].plot(times,d_ref,linestyle = 'dashed')
axs[0, 1].step(times,u)
axs[0, 1].step(times,ui)
axs[0, 1].set_ylabel("Ângulo de esterçamento[rad]")
axs[0, 1].set_xlabel("t[s]")
axs[0, 1].legend()
axs[1, 1].plot(a,b,linestyle = 'dashed')
axs[1, 1].plot(x,y)
axs[1, 1].plot(xi,yi)
axs[1, 1].set_ylabel("y[m]")
axs[1, 1].set_xlabel("x[m]")
axs[1, 1].legend()
#axs[1,1].plot(np.linspace(0,Nsim+1,Nsim+1),avgt*1000, label = 'Tempo de processamento por iteração')
#axs[1, 1].set_ylabel("tempo[ms]")
#axs[1, 1].set_xlabel("iteração")
#axs[1,1].legend()
axs[2,1].set_visible(False)
fig.suptitle('Comparação LTV X LTI')
# print(mse)
# print(max_x)
plt.show()