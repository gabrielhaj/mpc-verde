from cmath import sqrt
from time import time

import mpctools as mpc
import matplotlib.pyplot as plt
from mpctools.tools import DiscreteSimulator
import numpy as np
from sympy import sqrt_mod
from simulation_code import simulate
import casadi as ca
import pandas as pd
import mpctools.plots as mpcplots


Delta = 0.2; #sampling time
Nt = 10; #horizon step 
rob_dim = 0.3
Nx = 3
Nu = 2
Q_x = 1
Q_y = 5
Q_theta = 0.1
R1 = 0.5
R2 = 0.05
Q = np.eye(Nx)
Q[0,0] = Q_x
Q[1,1] = Q_y
Q[2,2] = Q_theta
R = np.eye(Nu)
R[0,0] = R1
R[1,1] = R2

v_max = 1;
v_min = -v_max;
omega_max = np.pi/4;
omega_min = -omega_max;


#model
def ode(x,u):
    dxdt = [
        (u[0]*np.cos(x[2])),
        (u[0]*np.sin(x[2])),
        u[1]]
    return np.array(dxdt)
 
#simulator
model = mpc.DiscreteSimulator(ode,Delta,[Nx,Nu], ["x","u"])

#casadi function for rk4 discretization
ode_rk4_casadi = mpc.getCasadiFunc(ode,[Nx,Nu],["x","u"], funcname = "F", rk4 = True, Delta = Delta, M = 1)

#target
p = np.zeros((Nt,(Nx+Nu))) #Matrix with sizes [Horizon lenght]x[States+Controls length]

#stage cost
def lfunc(x,u,p):
    return (x-p[:Nx]).T@Q@(x-p[:Nx]) + (u-p[Nx:Nu+Nx]).T@R@(u-p[Nx:Nu+Nx])
largs = ["x","u","p"]     
l = mpc.getCasadiFunc(lfunc,[Nx,Nu,(Nx+Nu)],largs, funcname = "l") 
funcargs = {"l": largs}

#bound on u
lb = {"u" : np.array([v_min,omega_min]),
    "x": np.array([-20,-2,-ca.inf])}
ub = {"u" : np.array([v_max,omega_max]),
    "x": np.array([20,2,ca.inf])}

#Make optimizers
x0 = np.array([0,0,0])
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Nx+Nu)}
solver = mpc.nmpc(f=ode_rk4_casadi, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True)

#simulate
Nsim = 500
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
pred = []
upred = []
avgt = np.array([0])
main_loop = time()  # return time in sec
par = np.ones((Nx+Nu,Nt,Nsim))

#Building the parameter 3d-matrix
#with sizes [Parameters lenght]x[Horizon length]x[Simulation lenght]
for t in range(Nsim):
    current_t = times[t]
    for k in range(Nt):
            t_predict = current_t + times[k]
            p[k,0] = np.cos(0.1*t_predict) #x_ref
            p[k,1] = np.sin(0.1*t_predict) #y_ref
            p[k,2] = np.pi/2 + 0.1*t_predict #theta_ref
            p[k,3] = 1 #u_ref
            p[k,4] = 1 #omega_ref
            par[:,k,t] = p[k,:]

            
for t in range(Nsim):
    # Fix initial state.
    
    # Solve nlp.
    t1 = time()
    for k in range(Nt):
        solver.par["p",k] = par[:,k,t] #trajectory is the parameter "p"  
    solver.solve()
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
    solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1])

    u[t,:] = np.array(solver.var["u",0,:]).flatten()
    
    #predicted trajectories
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    x[t+1,:] = model.sim(x[t,:],u[t,:])
    t2 = time()
    avgt = np.vstack((
        avgt,
        t2-t1
    ))
main_loop_time = time()
pred2 = []
pred3 = []    
for i in pred:
    pred2 = []
    for j in i:
        temp = np.array(j)
        pred2 += [temp]
    pred3 += [pred2]

pred3 = np.array(pred3)[:,:,:,0]


upred2 = []
upred3 = []    
for i in upred:
    upred2 = []
    for j in i:
        temp = np.array(j)
        upred2 += [temp]
    upred3 += [upred2]

upred3 = np.array(upred3)[:,:,:,0]
total_time = main_loop_time - main_loop
avg = np.array(avgt).mean() * 1000
table = [total_time,avg]
print('\n\n')
print('Total time: ', main_loop_time - main_loop)

# simulate(pred3.T, upred3.T, times, Delta, Nt,
#               np.array([0, 0, 0, 12, 1, 0]), save=True)
# fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
# plt.show()
#mpc.plots.showandsave(fig,"my_mpctools.pdf")
#simulate(pred3.T, upred3.T, times, Delta, Nt,
#             np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
#fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
# plt.show()
# mpc.plots.showandsave(fig,"Circular_Trajectory_Trakcing.pdf")

plt.plot(x[:,0],x[:,1], label = 'actual trajectory')
plt.plot(np.cos(0.1*times),np.sin(0.1*times), linestyle=('dashed'),label ='reference trajectory')
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend(loc='upper left')
plt.savefig('circular1.png')
plt.show()


# fig, axs = plt.subplots(4, 2)
# axs[0, 0].plot(times,x[:,0])
# axs[0, 0].set_ylabel("x[m]")
# axs[0, 0].set_xlabel("t[s]")
# axs[1, 0].plot(times,x[:,1])
# axs[1, 0].set_ylabel("y[m]")
# axs[1, 0].set_xlabel("t[s]")
# axs[0, 1].step(times,np.append(u[:,0],u[-1,0]))
# axs[0, 1].set_ylabel("v[m/s]")
# axs[0, 1].set_xlabel("t[s]")
# axs[2, 0].plot(times,x[:,2])
# axs[2, 0].set_ylabel("theta[rad]")
# axs[2, 0].set_xlabel("t[s]")
# axs[1,1].step(times,np.append(u[:,1],u[-1,1]))
# axs[1, 1].set_ylabel("w[rad/s]")
# axs[1, 1].set_xlabel("t[s]")
# axs[2,1].set_visible(False)
# axs[3,0].set_visible(False)
# axs[3,1].set_visible(False)


# plt.show()
# mpcplots.showandsave(fig, "Trajectoru.png")
