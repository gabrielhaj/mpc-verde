##tentativa de rodar esse exemplo do matlab
#https://www.mathworks.com/help/mpc/ug/control-of-an-inverted-pendulum-on-a-cart.html

import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt

#model from matlab example
Ac =  np.array([[0,0,0,0],[1,-10,0,-20],[0,9.81,0,39.24],[0,0,1,0]])
Ac = Ac.T

Bc = np.array([[0],[1],[0],[2]])


Nx = 4 #states size
Nu = 1 #controls size

#discretization

T = 0.5 #sampling time
Nt = 50 #prediction horizon
(A,B) = mpc.util.c2d(Ac,Bc,T) #continuos to discrete 
def ffunc(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
    #return mpc.mtimes(Ac,x) + mpc.mtimes(Bc,u)
f = mpc.getCasadiFunc(ffunc, [Nx,Nu], ["x","u"], "f")
#making ffunc as casadi function f

#bound on u
#umax = 2
umax = 200
lb = {"u": np.array([-umax])} #lower bound for control
#lb = {"x": np.array([-200,-500,-np.pi*100,-np.pi*100])}
ub = {"u": np.array([umax])} #upper bound for control
#ub = {"x": np.array([200,500,np.pi*100,np.pi*100])}
xt = np.array([10,0,0,0]) #reference values for the states
#Q and R matrices
#Q = np.diag([1.2,0,1,0])
Q = np.diag([120,0,100,0]) #weight matrix for states
R1 = 1 #weight value for control
#R1 = 100

def lfunc(x,u):
    [x1,x2,x3,x4] = x[:]
    return Q[0,0]*(x1-xt[0])**2 + Q[2,2]*(x3)**2 + R1*u**2
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x", "u"], "l")
#stage cost function l

#intial conditons
x0 = np.array([0,0,0,0])
#size of the states, controls and prediction horizon
N = {"x": Nx, "u" : Nu, "t" : Nt}

#simulation
solver = mpc.nmpc(f,l,N,x0,lb,ub,verbosity = 0, isQP = True)
#isQp = True means quadratic programming problem a.k.a linear and quadratic cost
nsim = 20 #number of simulations
t = np.arange(nsim+1)*T #time array
xcl = np.zeros((Nx,nsim+1)) #states record
xcl[:,0] = x0
ucl = np.zeros((Nu,nsim)) #control record
for k in range(nsim): #main loop
    solver.fixvar("x",0,x0) #setting initial value for state as always x0 at time 0
    sol = mpc.callSolver(solver) #solving
    print("Iteration %d Status: %s" % (k, sol["status"]))
    xcl[:,k] = sol["x"][0,:] #recording states
    ucl[:,k] = sol["u"][0,:] #recording optimal control
    x0 = ffunc(x0, ucl[:,k]) # Update x0.
xcl[:,nsim] = x0 # Store final state.

fig = mpc.plots.mpcplot(xcl,ucl,t,
                        timefirst=False, xnames = ["x", "x_dot","theta","theta_dot"], unames = ["Force"])
plt.show()
#mpcplots.showandsave(fig, "invertedpendulummpctools.pdf")




