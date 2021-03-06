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

#Numerical time shift function
def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0    

sim_time = 20
x_init = 0
y_init = 0
theta_init = 0
x_target = 10
y_target = 10
theta_target = 0
T = 0.2; #sampling time
N = 10; #horizon step
rob_dim = 0.3;
v_max = 1;
v_min = -v_max;
omega_max = pi/4;
omega_min = -omega_max;
Q_x = 1;
Q_y = 5;
Q_theta = 0.1;
R1 = 0.5;
R2 = 0.05;

#defining symbolic variables and building the state and input vectors
x = SX.sym('x');
y = SX.sym('y');
theta = SX.sym('theta');

states = ca.vertcat(
    x,
    y,
    theta
);
n_states = states.numel();

v = SX.sym('v');
omega = SX.sym('omega');
controls = ca.vertcat(
    v,
    omega
)
n_controls = controls.numel();

#Defining the right hand side of the equation
rhs = ca.vertcat(
    v*cos(theta),
    v*sin(theta),
    omega
);
#Defining the parameters, controls and states symbolic matrices 
P = ca.SX.sym('P', n_states + n_states) #The parameters are the initial state and the final state 
U = ca.SX.sym('U', n_controls, N)
X = ca.SX.sym('X',n_states,N+1)

#Defining the symbolic function that relates states,controls and rhs
f = ca.Function('f',[states,controls],[rhs],['x','u'],['rhs'])


#computation of the symbolic solution
X[:,0]= P[:n_states]
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    f_value = f(st,con)
    st_next = st + f_value*T
    X[:,k+1] = st_next

#ff function gives the predicted horizon for a given initial state 
#and controls actions over the horizon    
ff = ca.Function('ff',[U,P],[X])

Q = ca.diagcat(Q_x,Q_y,Q_theta)
R = ca.diagcat(R1,R2)

obj = 0;
#computation of the objective function
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    obj = obj + ((st-P[n_states:]).T @ Q @ (st-P[n_states:]) + con.T @ R @ con)
#defining the constrain vector
g = ca.reshape(X,(N+1)*n_states,1)
#defing the optimization variables
OPT_variables = ca.vertcat(
    U.reshape((-1, 1))
)

#setting the nlp
nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
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

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

#defining the constrains
lbx = ca.DM.zeros((n_controls*N, 1))
ubx = ca.DM.zeros((n_controls*N, 1))
lbx[0: n_controls*N: n_controls] = v_min
lbx[1: n_controls*N: n_controls] = omega_min
ubx[0: n_controls*N: n_controls] = v_max
ubx[1: n_controls*N: n_controls] = omega_max
args = {
    'lbg': -ca.inf,
    'ubg': ca.inf,
    #'lbg': -30,
    #'ubg': 30,
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state
t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls, N))  # initial control = [0,0,0,] (n_controls x N)
X0 = ca.repmat(state_init, 1, N+1)         # initial state full [state_init,state_init...N*]
#original era N+1
mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])

###############################################################################

#Solving the nlp inside the loop
if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * T < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.reshape(u0.T, n_controls*N, 1)
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        u = ca.reshape(sol['x'], n_controls, N)
        X0 = ff(u,args['p'])
        #Building the states horizon history
        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))
        #Building the control horizon history
        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))
        t0, state_init, u0 = shift_timestep(T, t0, state_init, u, f)

        X0 = ca.horzcat(
            X0[:, :],
            ca.reshape(X0[:, -1], -1, 1)
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
    print('Total time: ', table[1])
    print('avg iteration time: ',table[2] , 'ms')
    print('final error: ', table[0])

simulate(cat_states, cat_controls, times, T, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)

q = cat_states[:,0,:].T
w = cat_controls.reshape((85,2))
w = w[1:,:]
z = np.append(0,np.arange(0,round(t[-1][0],1),T))
z = np.append(z,16.6)
fig = mpc.plots.mpcplot(q,w,z, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])
plt.show()
mpc.plots.showandsave(fig,"my_mpc_code.pdf")

# p = pd.read_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx") 

# p["single_shooting_euler"] = table

# p.to_excel("mpc-euler-rk4-multipleshooting-tools-comparison.xlsx", sheet_name="Data", merge_cells=False)

