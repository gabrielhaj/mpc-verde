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
xref = g["x"]
yref = g["y"]
vref = g["uref"]

Delta = 0.05; #sampling time
Nt = 100; #time horizon 
Ntu = 100 #Control horizon
Nx = 4 # 4 states (lateral position, yaw angle, lateral velocity and angular velocity)
Nu = 1 #1 control (steering angle)
Q_y = 1
Q_phi = 1
Q_v = 1
Q_r = 1
R1 = 1
Q = np.eye(Nx)
Q[0,0] = Q_y
Q[1,1] = Q_phi
Q[2,2] = Q_v
Q[3,3] = Q_r
R = R1
delta_max = 20;
delta_min = -delta_max;

m = 1200
a =1.5
b = 2
Ca = 55000
Jz = 1350
ar = -23.55
br = 61.99
uref = 20

A33 = -4*Ca/(m*uref)
A34 = (2*Ca*(b-a)/m*uref) - uref
A43 = 2*Ca*((b-a)/(Jz*uref))
A44 = -2*Ca*(a**2 + b**2)/(Jz*uref)
B31 = 2*Ca/m
B41 = 2*Ca*a/Jz

#model
Ac = np.array(([0,uref,1,0],[0,0,0,1],[0,0,A33,A34],[0,0,A43,A44]))
Bc = np.array(([0],[0],[B31],[B41]))

def fcfunc(x,u):
    return mpc.mtimes(Ac,x) + mpc.mtimes(Bc,u)
fc = mpc.getCasadiFunc(fcfunc, [Nx,Nu], ["x","u"], "f") 

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

#Applying the horizon control limit
Dulb = np.tile(-np.inf,(Ntu,1)) #enabling changes until horizon control limit
Duub = np.tile(np.inf,(Ntu,1)) 
Dub = np.tile(0,(Nt-Ntu,1)) #forcing the rest of the horizon to stay the same (Move blocking)

lb = {"u": np.array([delta_min]),
    "Du": np.vstack((Dulb,Dub))} #lower bound for control

ub = {"u": np.array([delta_max]),
    "Du": np.vstack((Duub,Dub))} #upper bound for control


#Make optimizers

x0 = np.array([0,0,0,0])
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Nx+Nu)}
solver = mpc.nmpc(f, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True, uprev = np.array([0]))
# u = solver.varsym["u"]
# x = solver.varsym["x"]

#simulator

model = mpc.DiscreteSimulator(fc,Delta,[Nx,Nu], ["x","u"])

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
par = np.zeros((Nx+Nu,Nt,Nsim))
veclim = 499
for t in range(Nsim):
    for k in range(Nt):
            if(t+k > veclim):
                p[k,0] = yref[veclim] #y_ref
                p[k,1] = np.arctan2(yref[veclim],xref[veclim]) #phi_ref
                p[k,2] = vref[veclim]
            else:
                p[k,0] = yref[k+t] #y_ref
                p[k,1] = np.arctan2(yref[k+t],xref[k+t]) #phi_ref
                p[k,2] = vref[k+t] #vref
            if(t+k < 2):
                phi_ref_plus = np.arctan2(yref[k+1+t],xref[k+1+t])
                v_ref_plus = vref[k+t+1]
                v_dot = (v_ref_plus - p[k,2])/Delta #vrefdot
                p[k,3] = (phi_ref_plus - p[k,1])/Delta  #r_ref = (phi_ref+ - phi_ref)/Delta 
                p[k,4] = (v_dot - A33*p[k,2]-A34*p[k,3])/B31
            elif(t+k > 497):
                p[k,3] = (p[k,1] - par[1,k,t-1])/Delta  #r_ref = (phi_ref - phi_ref-)/Delta 
                p[k,4] = (v_dot - A33*p[k,2]-A34*p[k,3])/B31
            else:
                phi_ref_plus = np.arctan2(yref[k+1+t],xref[k+1+t])
                p[k,3] = (phi_ref_plus - par[1,k,t-1])/(2*Delta)  #r_ref = (phi_ref+ - phi_ref-)/2*Delta
                #delta_ref = (d/dx(r_ref) - ar*r_ref)/br
                p[k,4] = (v_dot - A33*p[k,2]-A34*p[k,3])/B31 
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
pred4 = np.ones((Nsim,Nt+1,Nx))
for n in range(Nsim):
    for k in range(Nt):
        pred4[n,k,0] = uref*Delta*(n+k)
        pred4[n,k,1] = pred3[n,k,0]
        pred4[n,k,2] = pred3[n,k,1]


# simulate(pred4.T, upred3.T, times, Delta, Nt,
#               np.array([0, 0, 0, 50, 50, 0]), save=True)

# a=0

fig = mpc.plots.mpcplot(x,u,times, xnames = ["Lateral Positon","Yaw Angle","Lateral velocity", "Angular velocity"], unames= ["Steering angle"])
plt.show()
#mpc.plots.showandsave(fig,"my_mpctools.pdf")
# simulate(pred3.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
#fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])

