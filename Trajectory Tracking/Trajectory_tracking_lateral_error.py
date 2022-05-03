from time import time
import mpctools as mpc
import matplotlib.pyplot as plt
from mpctools.tools import DiscreteSimulator
import numpy as np
from simulation_code import simulate
import casadi as ca
import pandas as pd


g = pd.read_csv("lane_change.csv")
a = g["x"]
b = g["y"]
c = g["uref"]

Delta = 0.05; #sampling time
Nt = 20; #Time horizon  
Ntu = 3; #Control horizon
Nx = 3 # 3 states (lateral position, yaw angle and angular position)
Nu = 1 #1 control (steering angle)
Q_y = 20
Q_phi = 1
Q_r = 1
R1 = 1
Q = np.eye(Nx)
Q[0,0] = Q_y
Q[1,1] = Q_phi
Q[2,2] = Q_r
R = R1
delta_max = 20;
delta_min = -delta_max;

ar = -23.55
br = 61.99
uref = np.mean(c)
#model
Ac = np.array(([0,uref,0],[0,0,1],[0,0,ar]))
Bc = np.array(([0],[0],[br]))

(A,B) = mpc.util.c2d(Ac,Bc,Delta) #continuos to discrete
def simf(x,u):
    return mpc.mtimes(Ac,x) + mpc.mtimes(Bc,u)
fsim = mpc.getCasadiFunc(simf,[Nx,Nu],["x","u"],"fsim")    
def ffunc(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
    #return A*x + B*u
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

x0 = np.array([0,0,0])
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Nx+Nu)}
solver = mpc.nmpc(f, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True,uprev = np.array([0]))
# u = solver.varsym["u"]
# x = solver.varsym["x"]

#simulator

model = mpc.DiscreteSimulator(fsim,Delta,[Nx,Nu], ["x","u"])

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
pred4 = np.ones((Nsim,Nt+1,3))
for n in range(Nsim):
    for k in range(Nt):
        pred4[n,k,0] = uref*Delta*(n+k)
        pred4[n,k,1] = pred3[n,k,0]
        pred4[n,k,2] = pred3[n,k,1]
xp = []
yp = []
for n in range(Nsim):
    xp += [uref*Delta*n]
    yp += [x[n,0]]
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(times,x[:,0], label ='actual')
axs[0, 0].plot(times[:-1],par[0,0,:], label ='reference')
axs[0, 0].set_ylabel("Lateral position")
axs[0, 0].set_xlabel("t[s]")
axs[0, 0].legend()
axs[1, 0].plot(times,x[:,1],label ='actual')
axs[1, 0].plot(times[:-1],par[1,0,:],label ='reference')
axs[1, 0].set_ylabel("Yaw angle")
axs[1, 0].set_xlabel("t[s]")
axs[1, 0].legend()
axs[1, 1].plot(times,x[:,2], label ='actual')
axs[1, 1].plot(times[:-1],par[2,0,:], label ='reference')
axs[1, 1].set_ylabel("Yaw rate")
axs[1, 1].set_xlabel("t[s]")
axs[1, 1].legend()
axs[0, 1].step(times[:-1],u, label ='actual')
axs[0, 1].plot(times[:-1],par[3,0,:], label ='reference')
axs[0, 1].set_ylabel("Steering angle")
axs[0, 1].set_xlabel("t[s]")
axs[0, 1].legend()
axs[2, 0].plot(xp,yp, label = 'actual trajectory')
axs[2, 0].plot(a,b, label = 'reference trajectory')
axs[2, 0].set_ylabel("y[m]")
axs[2, 0].set_xlabel("x[m]")
axs[2, 0].legend()
axs[2,1].set_visible(False)

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

# axs[0, 1].step(times,np.append(u[:,0],u[-1,0]))
# axs[0, 1].set_ylabel("v[m/s]")
# axs[0, 1].set_xlabel("t[s]")
# axs[2, 0].plot(times,x[:,2])
# axs[2, 0].set_ylabel("theta[rad]")
# axs[2, 0].set_xlabel("t[s]")    
#plt.plot(times,x[:,0])
#plt.plot(times[:-1],par[0,0,:])
plt.show()
plt.savefig(fname = 'LTI.png',bbox_inches='tight')
#simulate(pred4.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, 50, 50, 0]), save=False)

#fig = mpc.plots.mpcplot(x,u,times, xnames = ["Lateral Positon","Yaw Angle", "Angular velocity"], unames= ["Steering angle"])
#plt.show()
#mpc.plots.showandsave(fig,"my_mpctools.pdf")
# simulate(pred3.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
#fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])

#mpc.plots.showandsave(fig,"verde_trajectory_tracking.pdf")
