from time import time
import mpctools as mpc
import matplotlib.pyplot as plt
from mpctools.tools import DiscreteSimulator
import numpy as np
from simulation_code import simulate
import casadi as ca
import pandas as pd
Delta = 0.2; #sampling time old "T"
Nt = 10; #horizon step old "N"
rob_dim = 0.3
Nx = 3
Nu = 2
x_target = 10
y_target = 10
theta_target = 0
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
goal = np.array([x_target,y_target,theta_target])
#stage cost
def lfunc(x,u):
    return (x-goal).T@Q@(x-goal) + u.T@u
l = mpc.getCasadiFunc(lfunc,[Nx,Nu],["x","u"], funcname = "l")   

#bound on u
lb = {"u" : np.array([v_min,omega_min])}
ub = {"u" : np.array([v_max,omega_max])}

#Make optimizers
x0 = np.array([0,0,0])
N = {"x":Nx, "u":Nu, "t": Nt}
solver = mpc.nmpc(f=ode_rk4_casadi, N = N, verbosity=0, l=l,x0 =x0,lb = lb, ub = ub)

#main loop
Nsim = 150
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
pred = []
upred = []
avgt = np.array([0])
main_loop = time()  # return time in sec
for t in range(Nsim):
    if(ca.norm_2(x[t,:] - goal) < 1e-1):
        break
    # Fix initial state.   
    
    # Solve nlp.
    t1 = time()
    solver.solve()
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
    #solver.saveguess() #salva o ultimo resultado como chute para a próxima resolução
    #Define a variável inicial a partir da ultima solução
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
ss_error = ca.norm_2(solver.var["x",1] - np.array([x_target,y_target,theta_target]))

total_time = main_loop_time - main_loop
avg = np.array(avgt).mean() * 1000
table = [ss_error,total_time,avg]
print('\n\n')
print('Total time: ', main_loop_time - main_loop)
print('final error: ', ss_error)

#simulate(pred3.T, upred3.T, times, Delta, Nt,
#              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
fig = mpc.plots.mpcplot(x[:t],u[:t-1],times[:t], xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
mpc.plots.showandsave(fig,"multipleshootingmpctools.pdf")

df = pd.DataFrame({
    "x": x[:t,0],
    "y": x[:t,1],
    "theta": x[:t,2],
    "v": u[:t,0],
    "w": u[:t,1],
    "t": times[:t]
})

df.to_excel("3exemplo.xlsx", sheet_name="Sheet1")

# p = pd.read_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx") 

# p["mpctools"] = table

# p.to_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx", sheet_name="Data", merge_cells=False) 