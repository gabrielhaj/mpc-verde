from time import time
import casadi as ca
import numpy as np
from casadi import *
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

from simulation_code import simulate
import mpctools as mpc
from mpctools.tools import DiscreteSimulator
import pandas as pd


def DM2Arr(dm):
    return np.array(dm.full())
# Original shift function
# def shift_timestep(step_horizon, t0, state_init, u, f):
#     f_value = f(state_init, u[:, 0])
#     next_state = ca.DM.full(state_init + (step_horizon * f_value))

#     t0 = t0 + step_horizon
#     u0 = ca.horzcat(
#         u[:, 1:],
#         ca.reshape(u[:, -1], -1, 1)
#     )

#     return t0, next_state, u0    


sim_time = 20
T = 0.2; #sampling time
N = 10; #horizon step
rob_dim = 0.3

x_init = 0
y_init = 0
theta_init = 0
x_target = 10
y_target = 10
theta_target = 0

v_max = 1;
v_min = -v_max;
omega_max = pi/4;
omega_min = -omega_max;

#model states variables
x = SX.sym('x')
y = SX.sym('y')
theta = SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

#model control variables
v = SX.sym('v')
omega = SX.sym('omega')
controls = ca.vertcat(
    v,
    omega
)
n_controls = controls.numel()

#right hand side equation
rhs = ca.vertcat(
    v*cos(theta),
    v*sin(theta),
    omega
)
#parameters (first n_states: initial values, last n_states: reference)
P = ca.SX.sym('P', n_states + n_states)
Po = P

#weights matrices
Q_x = 1
Q_y = 5
Q_theta = 0.1
R1 = 0.5
R2 = 0.05
Q = ca.diagcat(Q_x,Q_y,Q_theta)
R = ca.diagcat(R1,R2)

#obective term
L = (states-P[n_states:]).T @ Q @(states-P[n_states:]) + controls.T @ R @ controls

#control matrix
U = ca.SX.sym('U', n_controls)

#states matrix
X = ca.SX.sym('X',n_states,N+1)

#f = ca.Function('f',[states,controls],[rhs,L])
f = ca.Function('f',[states,controls,P],[rhs,L])

X0 = P[:n_states]
X = X0
Q = 0
M = 4
DT = T/M
for j in range(M):
    # k1, k1_q = f(X, U)
    # k2, k2_q = f(X + DT/2 * k1, U)
    # k3, k3_q = f(X + DT/2 * k2, U)
    # k4, k4_q = f(X + DT * k3, U)
    k1, k1_q = f(X, U, P)
    k2, k2_q = f(X + DT/2 * k1, U, P)
    k3, k3_q = f(X + DT/2 * k2, U, P)
    k4, k4_q = f(X + DT * k3, U, P)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [P, U], [X, Q],['x0','p'],['xf','qf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []
Xk = P[:n_states] #initial conditions

#"lifting" initial conditions
for a in range(3):
    Xa = ca.SX.sym('X_' + str(a))
    w += [Xa]
    g += [Xk[a]-Xa]
lbw += [-ca.inf, -ca.inf, -ca.inf]
ubw += [ca.inf, ca.inf, ca.inf]
w0 += [0, 0, 0]
lbg +=[0,0,0]
ubg +=[0,0,0]

# Formulate the NLP
e = 0
f = 3
for k in range(N):
    # New NLP variable for the control
    for a in range(e,e+2):
        Uk = ca.SX.sym('U_' + str(a))    
        w += [Uk]
        if(a == e):
            lbw += [v_min]
            ubw += [v_max]
            w0 += [0]
        else:
            lbw += [omega_min]
            ubw += [omega_max]
            w0 += [0]
        
    Uk = ca.vertcat(w[e+f],w[e+f+1]) #concat velocity "w[e+3]" and angular velocity"w[e+4]""
    # Integrate till the end of the interval
    Fk = F(x0=ca.vertcat(Xk,P[n_states:]), p=Uk)
    
    Xk_end = Fk['xf']
   
    J=J+Fk['qf']

    # New NLP variable for state at end of interval
    for a in range(f,f+3):
        X = ca.SX.sym('X_' + str(a))    
        w += [X]
    lbw += [-ca.inf, -ca.inf, -ca.inf]
    ubw += [ca.inf, ca.inf, ca.inf]
    w0  += [0, 0, 0]
    Xk = ca.vertcat(w[e+f+2],w[e+f+3],w[e+f+4])
    # Add equality constraint
    for a in range(n_states):
        g   += [Xk_end[a]-Xk[a]]
    lbg += [0, 0, 0]
    ubg += [0, 0, 0]

    e = e + 2
    f = f + 3

# Create an NLP solver
prob = {
    'f': J,
    'x': ca.vertcat(*w),
    'g': ca.vertcat(*g),
    #'p': Po
    'p': P
}
opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}
solver = ca.nlpsol('solver', 'ipopt', prob, opts);

args = {
    'lbg': lbg,
    'ubg': ubg,
    #'lbg': -30,
    #'ubg': 30,
    'lbx': lbw,
    'ubx': ubw
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state
t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])


###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * T < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = w0

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'] ,
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        u = sol['x'][n_states:n_states+2]
        e = n_states + 5
        for k in range(N-1):
            uk = ca.vertcat(sol['x'][e],sol['x'][e+1])
            u = ca.vertcat(u,uk)
            e = e + 5
        X0 = sol['x'][:n_states]
        e = n_states + 2
        for k in range(N):
            xk = ca.vertcat(sol['x'][e],sol['x'][e+1],sol['x'][e+2])
            X0 = ca.vertcat(X0,xk)
            e = e + 5                     
        u = ca.reshape(u,n_controls,N)
        X0 = ca.reshape(X0,n_states,N+1)
        
        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))
        # applying the time shift
        t0 = t0 + T
        state_init = F(args['p'],u[:,0])[0] 
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
        w0 = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    
    total_time = main_loop_time - main_loop
    avg = np.array(times).mean() * 1000
    table = [ss_error,total_time,avg]

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

# simulate(cat_states, cat_controls, times, T, N,
#              np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)

q = cat_states[:,0,:].T
w = cat_controls.reshape((85,2))
w = w[1:,:]
z = np.append(0,np.arange(0,round(t[-1][0],1),T))
z = np.append(z,16.6)
# fig = mpc.plots.mpcplot(q,w,z, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
# plt.show()
# mpc.plots.showandsave(fig,"multiple_shooting_casadi.pdf")             

df = pd.DataFrame({
    "x": q[:,0],
    "y": q[:,1],
    "theta": q[:,2],
    "v": np.append(w[:,0],w[-1,0]),
    "w": np.append(w[:,1],w[-1,1]),
    "t": z
})

df.to_excel("1exemplo.xlsx", sheet_name="Sheet1")


# p = pd.read_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx") 

# p["multiple_shooting"] = table
# name = ["Erro quadrático","Tempo total", "Tempo médio"]
# p["Data"] = name
# p.to_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx", sheet_name="Data", merge_cells=False) 