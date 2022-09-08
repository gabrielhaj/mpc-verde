from ast import Del
from cmath import sqrt
from time import time
from tkinter import Y
import mpctools as mpc
import matplotlib.pyplot as plt
from mpctools.tools import DiscreteSimulator
import numpy as np
from sympy import sqrt_mod
from simulation_code import simulate
import casadi as ca
import pandas as pd

g = pd.read_csv("lane_change.csv")
xtraj = g["x"]
ytraj = g["y"]
vdes = g["uref"]

L = 3.5
Delta = 0.05; #sampling time
Nt = 20; #time horizon 
Nx = 3 # 3 states (lateral error(ye=y-yt), yaw error(phie = phi-phit) and velocity)
Nu = 2 #1 control (steering angle(delta) and aceleration)
Np = 4 #4 parameters(lateral trajectory(yt), yaw angle(phit), trajectory curvature(kappat), desired velocity(vdes))
lambda2 = 1.75 #lambda2 (ye)
lambda3 = 2.5 #lambda3 (phie)
lambda1 = 2.5 #lambda1 (v-vdes)
lambda4 = 0.4 #a
lambda5 = 10 #z
R = 10 #(z = tan(delta)-L*kappa)
delta_max = 0.384;
delta_min = -delta_max;
a_max = 2
a_min = -a_max
delta_dot_max = 0.1225
delta_dot_min = -delta_dot_max

#target
p = np.zeros((Nt,(Np))) #Nt lines, Np columns

#stage cost
def lfunc(x,u,p):
    [y,phi,v] = x[:Nx]
    [delta,a] = u[:Nu]
    [yt,phit,kappat] = p[:Nx]
    [vdes] = p[Nx:Np]
    R = 1/kappat
    z = np.tan(delta) - L*kappat
    return (lambda1*(v-vdes)**2 + lambda2*(y-yt)**2 + lambda3*(phi-phit)**2 + lambda4*a**2 + lambda5*z**2)/(Nt+1)
largs = ["x","u","p"]     
l = mpc.getCasadiFunc(lfunc,[Nx,Nu,Np],largs, funcname = "l") 


#bounds
lb = {"u": np.array([delta_min,a_min]),
    "Du": np.array([delta_dot_min,-ca.inf])} #lower bound for control

ub = {"u": np.array([delta_max,a_max]),
    "Du": np.array([delta_dot_max,ca.inf])} #upper bound for control
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Np)}

#simulate

Nsim = 500
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = [0,0,0]
x0 = x[0,:]
u = np.zeros((Nsim,Nu))
pred = []
upred = []
avgt = np.array([0])
main_loop = time()  # return time in sec
par = np.zeros((Np,Nsim))

for t in range(Nsim):

    #trajectory
    for k in range(Nt):
            if(t+k > Nsim-1):
                p[k,0] = ytraj[Nsim-1] #y_ref
                p[k,1] = np.arctan2(ytraj[Nsim-1]-ytraj[Nsim-2],xtraj[Nsim-1]-xtraj[Nsim-2]) #phi_ref
            elif(t + k == 0):
                p[k,0] = ytraj[k+t] #y_ref
                p[k,1] = 0 #phi_ref
            else:
                p[k,0] = ytraj[k+t] #y_ref
                p[k,1] = np.arctan2(ytraj[k+t]-ytraj[k+t-1],xtraj[k+t]-xtraj[k+t-1]) #phi_ref
            if(t+k < 2):
                p[k,3] = 1
                p[k,2] = vdes[t+k]
            elif(t+k > Nsim-2):
                p[k,3] = p[k-1,3]
                p[k,2] = vdes[Nsim-1]
            else:
                ddx = (xtraj[k+t-1] -2*xtraj[k+t] + xtraj[k+t+1])/Delta**2
                ddy = (ytraj[k+t-1] -2*ytraj[k+t] + ytraj[k+t+1])/Delta**2    
                p[k,3] = np.linalg.norm(np.array([ddx,ddy]))
                p[k,2] = vdes[t+k]
    par[:,t] = p[0,:]
                
    #model
    def ode(x,u,p):
        [y,phi,v] = x[:Nx]
        [delta,a] = u[:Nu]
        [yt,phit,kappat] = p[:Nx]
        dxdt = [
            v*np.sin(phi-phit),
            v*(np.tan(delta/L)-(kappat/(1-(y-yt)*kappat))*np.cos(phi-phit)),
            a
        ]
        return np.array(dxdt)
    
    #simulator
    model = mpc.DiscreteSimulator(ode,Delta,[Nx,Nu,Np], ["x","u","p"])

    #casadi function for rk4 discretization
    ode_rk4_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Np],["x","u","p"], funcname = "F", rk4 = True, Delta = Delta, M = 1)
    funcargs = {"l": largs,"F": largs}
    
    #controlador
    solver = mpc.nmpc(ode_rk4_casadi, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True)
    
    # Solve nlp.
    t1 = time()
    for k in range(Nt): #passing parameters along the horizon control
        solver.par["p",k] = p[k,:] #trajectory is the parameter "p"  

    solver.solve() #solving
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
    solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1]) #x[k+1] <- x[k]
    u[t,:] = np.array(solver.var["u",0,:]).flatten()
    
    #predicted trajectories
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    x[t+1,:] = model.sim(x[t,:],u[t,:],p[0,:])
    x0 = x[t+1,:] #build vector x
    t2 = time()
    avgt = np.vstack((
        avgt,
        t2-t1
    ))
xz = []
yz = []
for n in range(Nsim):
    if(n == 0):
        xz += [0]
    else:        
        xz += [xz[n-1] + x[n,2]*np.cos(x[n,1])*Delta]
    yz += [x[n,0]]    
main_loop_time = time()
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(times,x[:,0], label ='actual')
axs[0, 0].plot(times[:-1],par[0,:], label ='reference')
axs[0, 0].set_ylabel("Lateral position")
axs[0, 0].set_xlabel("t[s]")
axs[0, 0].legend()
axs[1, 0].plot(times,x[:,1],label ='actual')
axs[1, 0].plot(times[:-1],par[1,:],label ='reference')
axs[1, 0].set_ylabel("Yaw angle")
axs[1, 0].set_xlabel("t[s]")
axs[1, 0].legend()
axs[2, 0].plot(times,x[:,2], label ='actual')
axs[2, 0].plot(times[:-1],par[2,:], label ='reference')
axs[2, 0].set_ylabel("Yaw rate")
axs[2, 0].set_xlabel("t[s]")
axs[2, 0].legend()
axs[0, 1].step(times[:-1],u[:,0], label ='actual')
axs[0, 1].set_ylabel("Steering angle")
axs[0, 1].set_xlabel("t[s]")
axs[0, 1].legend()
axs[1, 1].step(times[:-1],u[:,1], label ='actual')
axs[1, 1].set_ylabel("Aceleration")
axs[1, 1].set_xlabel("t[s]")
axs[1, 1].legend()
axs[2, 1].plot(xz,yz, label = 'actual trajectory')
axs[2, 1].plot(xtraj,ytraj, label = 'reference trajectory')
axs[2, 1].set_ylabel("y[m]")
axs[2, 1].set_xlabel("x[m]")
axs[2, 1].legend()
plt.show()
plt.savefig(fname = 'ltv.png', orientation = 'landscape')

