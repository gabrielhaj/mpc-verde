from time import time
import casadi as ca
import numpy as np
from casadi import *
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from mpc_code import Q_theta
from simulation_code import simulate
import mpctools as mpc
from mpctools.tools import DiscreteSimulator


def DM2Arr(dm):
    return np.array(dm.full())
# Original shift function
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
Xk = P[:n_states]
# Formulate the NLP
#Pk = SX([x_init,y_init,theta_init,x_target,y_target,theta_target])
e = 0
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
        
    Uk = ca.vertcat(w[e],w[e+1])
    # Integrate till the end of the interval
    Fk = F(x0=ca.vertcat(Xk,P[n_states:]), p=Uk)
    #Fk = F(x0=Pk, p=Uk)
    Xk = Fk['xf']
    X = ca.horzcat(X,ca.vertcat(Xk))
    #Pk[:n_states] = Xk
    #Po[:n_states] = Xk
    J=J+Fk['qf']

    # Add inequality constraint
    g += [Xk[0]]
    g += [Xk[1]]
    lbg += [-ca.inf]
    ubg += [ca.inf]

    e = e + 2

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
    'lbg': -ca.inf,
    'ubg': ca.inf,
    #'lbg': -30,
    #'ubg': 30,
    'lbx': lbw,
    'ubx': ubw
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state
t = ca.DM(t0)
X0 = state_init #aux variable
X1 = X0 #aux variable
mpc_iter = 0
cat_states = DM2Arr(ca.repmat(X0,1,N+1)) #variable to store the predicted states
cat_controls = DM2Arr(ca.DM.zeros((n_controls))) #variable to store the applied controls
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

        u = ca.reshape(sol['x'], n_controls, N)
        for k in range(N):
            X1 = F(ca.vertcat(X1,state_target),u[:,k])[0]
            X0 = ca.horzcat(X0,X1)
        
        
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
        # applying the
        t0 = t0 + T
        state_init = F(args['p'],u[:,0])[0] 
        X0 = state_init
        X1 = X0
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )
        w0 = ca.reshape(u0,n_controls*N,1)

        # print(X0)
        # X0 = ca.horzcat(
        #     X0[:, :],
        #     ca.reshape(X0[:, -1], -1, 1)
        # )

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

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

simulate(cat_states, cat_controls, times, T, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)

q = cat_states[:,0,:].T
w = cat_controls.reshape((85,2))
w = w[1:,:]
z = np.append(0,np.arange(0,round(t[-1][0],1),T))
z = np.append(z,16.6)
fig = mpc.plots.mpcplot(q,w,z)
plt.show()
mpc.plots.showandsave(fig,"my_mpc_code2.pdf")             