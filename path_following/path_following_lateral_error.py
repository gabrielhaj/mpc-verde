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


g = pd.read_csv("lane_change.csv")
a = g["x"]
b = g["y"]

Delta = 0.05; #sampling time
Nt = 10; #horizon step 
Nx = 3 # 3 states (lateral position, yaw angle and angular position)
Nu = 1 #1 control (steering angle)
Q_x = 1
Q_y = 1
Q_theta = 1
R1 = 1
Q = np.eye(Nx)
Q[0,0] = Q_x
Q[1,1] = Q_y
Q[2,2] = Q_theta
R = R1
delta_max =20;
delta_min = -delta_max;

ar = -23.55
br = 61.99
uref = 2
#model
Ac = np.array(([0,uref,0],[0,0,1],[0,0,ar]))
Bc = np.array(([0],[0],[br]))

(A,B) = mpc.util.c2d(Ac,Bc,Delta) #continuos to discrete

def ffunc(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
f = mpc.getCasadiFunc(ffunc, [Nx,Nu], ["x","u"], "f")
 

#target
p = np.zeros((Nt,(Nx+Nu))) #Nt lines, Nx+Nu columms

#stage cost
def lfunc(x,u,p):
    return (x-p[:Nx]).T@Q@(x-p[:Nx]) + R*(u-p[Nx:Nu+Nx])**2
largs = ["x","u","p"]     
l = mpc.getCasadiFunc(lfunc,[Nx,Nu,(Nx+Nu)],largs, funcname = "l") 
funcargs = {"l": largs}

#bound on delta
lb = {"u" : np.array([delta_min])}
ub = {"u" : np.array([delta_max])}

#Make optimizers

x0 = np.array([0,0,0])
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Nx+Nu)}
solver = mpc.nmpc(f, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True)
# u = solver.varsym["u"]
# x = solver.varsym["x"]

#simulator

model = mpc.DiscreteSimulator(f,Delta,[Nx,Nu], ["x","u"])

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
for t in range(Nsim):
    for k in range(Nt):
            if(t+k > 499):
                p[k,0] = b[499] #y_ref
                p[k,1] = np.arctan2(b[499],a[499]) #phi_ref
            else:
                p[k,0] = b[k+t] #y_ref
                p[k,1] = np.arctan2(b[k+t],a[k+t]) #phi_ref
            if(t+k < 2):
                phi_ref_plus = np.arctan2(b[k+1+t],a[k+1+t])
                phi_ref_plus2 = np.arctan2(b[k+2+t],a[k+2+t])
                p[k,2] = (phi_ref_plus - p[k,1])/Delta  #r_ref = (phi_ref+ - phi_ref)/Delta 
                p[k,3] = (((phi_ref_plus2 - 2*phi_ref_plus + p[k,1])/Delta**2) - ar*p[k,2])/br
            elif(t+k > 497):
                p[k,2] = (p[k,1] - par[1,k,t-1])/Delta  #r_ref = (phi_ref - phi_ref-)/Delta 
                p[k,3] = (((p[k,1] - 2*par[1,k,t-1] + par[1,k-1,t-1])/Delta**2) - ar*p[k,2])/br
            else:
                phi_ref_plus = np.arctan2(b[k+1+t],a[k+1+t])
                p[k,2] = (phi_ref_plus - par[1,k,t-1])/(2*Delta)  #r_ref = (phi_ref+ - phi_ref-)/2*Delta
                #delta_ref = (d/dx(r_ref) - ar*r_ref)/br
                p[k,3] = (((phi_ref_plus - 2*p[k,1] + par[1,k,t-1])/Delta**2) - ar*p[k,2])/br 
            par[:,k,t] = p[k,:]
for t in range(Nsim):
    # Fix initial state.
    
    #<<ENDCHUNK>>    
    
    # Solve nlp.
    t1 = time()
    for k in range(Nt):
        solver.par["p",k] = par[:,k,t] #trajectory is the parameter "p"  
    solver.solve()
    
    
    #<<ENDCHUNK>>    
    
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
pred4 = np.ones((500,11,3))
for n in range(Nsim):
    for k in range(Nt):
        pred4[n,k,0] = uref*Delta*(n+k)
        pred4[n,k,1] = pred3[n,k,0]
        pred4[n,k,2] = pred3[n,k,1]


simulate(pred4.T, upred3.T, times, Delta, Nt,
              np.array([0, 0, 0, 50, 50, 0]), save=True)

fig = mpc.plots.mpcplot(x,u,times, xnames = ["Lateral Positon","Yaw Angle", "Angular velocity"], unames= ["Steering angle"])
#plt.show()
#mpc.plots.showandsave(fig,"my_mpctools.pdf")
# simulate(pred3.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
#fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
mpc.plots.showandsave(fig,"verde_trajectory_tracking.pdf")
