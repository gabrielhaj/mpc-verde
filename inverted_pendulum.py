import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt
Ac =  np.array([[0,0,0,0],[1,-10,0,-20],[0,9.81,0,39.24],[0,0,1,0]])
Ac = Ac.T

Bc = np.array([[0],[1],[0],[2]])

Nx = 4
Nu = 1

#discretization

T = 0.5
Nt = 50
(A,B) = mpc.util.c2d(Ac,Bc,T)
def ffunc(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
    #return mpc.mtimes(Ac,x) + mpc.mtimes(Bc,u)
f = mpc.getCasadiFunc(ffunc, [Nx,Nu], ["x","u"], "f")    

#bound on u
#umax = 2
umax = 200
lb = {"u": np.array([-umax])}
#lb = {"x": np.array([-200,-500,-np.pi*100,-np.pi*100])}
ub = {"u": np.array([umax])}
#ub = {"x": np.array([200,500,np.pi*100,np.pi*100])}
xt = np.array([10,0,0,0])
#Q and R matrices
#Q = np.diag([1.2,0,1,0])
Q = np.diag([120,0,100,0])
R1 = 1
#R1 = 100

def lfunc(x,u):
    [x1,x2,x3,x4] = x[:]
    return Q[0,0]*(x1-xt[0])**2 + Q[2,2]*(x3)**2 + R1*u**2
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x", "u"], "l")

#intial conditons
x0 = np.array([0,0,0,0])
N = {"x": Nx, "u" : Nu, "t" : Nt}

#simulation
solver = mpc.nmpc(f,l,N,x0,lb,ub,verbosity = 0, isQP = True)
nsim = 100
t = np.arange(nsim+1)*T
xcl = np.zeros((Nx,nsim+1))
xcl[:,0] = x0
ucl = np.zeros((Nu,nsim))
for k in range(nsim):
    solver.fixvar("x",0,x0)
    sol = mpc.callSolver(solver)
    print("Iteration %d Status: %s" % (k, sol["status"]))
    xcl[:,k] = sol["x"][0,:]
    ucl[:,k] = sol["u"][0,:]
    x0 = ffunc(x0, ucl[:,k]) # Update x0.
xcl[:,nsim] = x0 # Store final state.

fig = mpc.plots.mpcplot(xcl,ucl,t,
                        timefirst=False)
plt.show()
#mpcplots.showandsave(fig, "invertedpendulummpctools.pdf")




