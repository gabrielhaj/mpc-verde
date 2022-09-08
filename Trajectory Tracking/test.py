from ast import Del
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
c = g["uref"]

L = 3.5
Delta = 0.05; #sampling time
Nt = 20; #time horizon 
Ntu = 3; #control horizon
Nx = 3 # 3 states (lateral position, yaw angle and angular velocity)
Nu = 1 #1 control (steering angle)
lambda2 = 1.75 #lambda2 (ye)
lambda3 = 2.5 #lambda3 (phie)
lambda1 = 2.5 #lambda1 (v-vdes)
R = 10 #(z = tan(delta)-L*kappa)
delta_max = 20;
delta_min = -delta_max;

ar = -23.55
br = 61.99
uref = 2

#target
p = np.zeros((Nt,(Nx+Nu))) #Nt lines, Nx+Nu columms


#Make optimizers

# u = solver.varsym["u"]
# x = solver.varsym["x"]

#stage cost
def lfunc(x,u,p):
    [y,phi,r] = x[:Nx]
    [delta] = u
    [yt,phit,kappat] = p[:Nx]
    [vdes] = p[Nx:Nx+Nu]
    R = 1/kappat
    vel = r*R
    z = np.tan(delta) - L*kappat
    return lambda2*(y-yt)**2 + lambda3*(phi-phit)**2 + lambda1*(vel-vdes)**2 + R*z**2
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
N = {"x":Nx, "u":Nu, "t": Nt,"p":(Nx+Nu)}
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
par = np.zeros((Nx+Nu,Nt,Nsim))
for t in range(Nsim):
    for k in range(Nt):
            if(t+k > Nsim-1):
                p[k,0] = b[Nsim-1] #y_ref
                p[k,1] = np.arctan2(b[Nsim-1]-b[Nsim-2],a[Nsim-1]-a[Nsim-2]) #phi_ref
            elif(t + k == 0):
                p[k,0] = b[k+t] #y_ref
                p[k,1] = 0 #phi_ref
            else:
                p[k,0] = b[k+t] #y_ref
                p[k,1] = np.arctan2(b[k+t]-b[k+t-1],a[k+t]-a[k+t-1]) #phi_ref
            if(t+k < 2):
                p[k,3] = 1
                p[k,2] = c[t+k]
            elif(t+k > Nsim-2):
                p[k,3] = p[k-1,3]
                p[k,2] = c[Nsim-1]
            else:
                ddx = (a[k+t-1] -2*a[k+t] + a[k+t+1])/Delta**2
                ddy = (b[k+t-1] -2*b[k+t] + b[k+t+1])/Delta**2    
                p[k,3] = np.linalg.norm(np.array([ddx,ddy]))
                p[k,2] = c[t+k]                
            par[:,k,t] = p[k,:]

for t in range(Nsim):
    #model
    Ac = np.array(([0,c[t],0],[0,0,1],[0,0,ar]))
    Bc = np.array(([0],[0],[br]))

    (A,B) = mpc.util.c2d(Ac,Bc,Delta) #continuos to discrete
    def simf(x,u):
        return mpc.mtimes(Ac,x) + mpc.mtimes(Bc,u)
    fsim = mpc.getCasadiFunc(simf,[Nx,Nu],["x","u"],"fsim")    
    def ffunc(x,u):
        return mpc.mtimes(A,x) + mpc.mtimes(B,u)
        #return A*x + B*u
    f = mpc.getCasadiFunc(ffunc, [Nx,Nu], ["x","u"], "f")
    
    #controlador
    solver = mpc.nmpc(f, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub,p = p,funcargs= funcargs, inferargs= True,uprev = np.array([0]))

    #simulator
    model = mpc.DiscreteSimulator(fsim,Delta,[Nx,Nu], ["x","u"])
    
    # Solve nlp.
    t1 = time()
    for k in range(Nt): #passing parameters along the horizon control
        solver.par["p",k] = par[:,k,t] #trajectory is the parameter "p"  

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
    x[t+1,:] = model.sim(x[t,:],u[t,:])
    x0 = x[t+1,:] #build vector x
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
xz = []
yz = []
for n in range(Nsim):
    if(n == 0):
        xz += [0]
    else:        
        xz += [xz[n-1] + c[n]*np.cos(x[n,1])*Delta]
    yz += [x[n,0]]
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
axs[1, 1].set_ylabel("Yaw rate")
axs[1, 1].set_xlabel("t[s]")
axs[1, 1].legend()
axs[0, 1].step(times[:-1],u, label ='actual')
axs[0, 1].set_ylabel("Steering angle")
axs[0, 1].set_xlabel("t[s]")
axs[0, 1].legend()
axs[2, 0].plot(xz,yz, label = 'actual trajectory')
axs[2, 0].plot(a,b, label = 'reference trajectory')
axs[2, 0].set_ylabel("y[m]")
axs[2, 0].set_xlabel("x[m]")
axs[2, 0].legend()
axs[2,1].set_visible(False)
plt.show()
plt.savefig(fname = 'ltv.png', orientation = 'landscape')

#simulate(pred4.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, 50, 50, 0]), save=False)

#fig = mpc.plots.mpcplot(x,u,times, xnames = ["Lateral Positon","Yaw Angle", "Angular velocity"], unames= ["Steering angle"])
#plt.show()
#mpc.plots.showandsave(fig,"my_mpctools.pdf")
# simulate(pred3.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
#fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])

#mpc.plots.showandsave(fig,"verde_trajectory_tracking.pdf")
