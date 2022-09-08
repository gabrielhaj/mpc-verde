from cmath import sqrt
from logging.handlers import NTEventLogHandler
from statistics import median_grouped
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
# a = a[:700]
# b = b[:700]
# c = c[:700]
#qq_y = [0.01,0.1, 1,10,100]
#horz  = [3,5,8,10,15]
horz = [5]
#horz = [g["uref"],np.ones(500)*g["uref"].mean()]
#fig, axs = plt.subplots(3, 1)
for Nt in horz:
#for Q_y in qq_y:
    dist = 0
    Delta = 0.05; #sampling time
    #Nt = 3
    Ntu = 1
    Nx = 3 # 3 states (lateral position, yaw angle and angular position)
    Nu = 1 #1 control (steering angle)
    Q_y = 10
    Q_theta = 1
    Q_thetaponto = 0
    R = 0.01
    Q = np.eye(Nx)
    Q[0,0] = Q_y
    Q[1,1] = Q_theta
    Q[2,2] = Q_thetaponto
    R1 = 0
    delta_max = 0.3491;
    delta_min = -delta_max;
    Nsim = a.size
    ar = -23.55
    br = 61.99

    #target
    p = np.zeros((Nt,(Nx+Nu))) #Nt lines, Nx+Nu columms


    #Make optimizers

    # u = solver.varsym["u"]
    # x = solver.varsym["x"]

    #stage cost
    def lfunc(x,u,p,du):
        return (x-p[:Nx]).T@Q@(x-p[:Nx]) + R*(u-p[Nx:Nu+Nx])**2 + R1*du**2
    largs = ["x","u","p","Du"]     
    l = mpc.getCasadiFunc(lfunc,[Nx,Nu,(Nx+Nu), Nu],largs, funcname = "l") 
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
    phi = np.zeros(Nsim)
    uref = np.zeros(Nsim)
    #simulate
    # for n in range(Nsim):
    #     if(n == 0):
    #         phi[n] = 0
    #     else:
    #         phi[n] = np.arctan2(b[n]-b[n-1],a[n]-a[n-1])
    #     uref[n] = c[n]*np.cos(phi[n])
    # uref = uref.mean()
    uref = c.mean()
    #model LTI
    Ac = np.array(([0, uref, 0], [0, 0, 1], [0, 0, ar]))
    #Ac = np.array(([0, c, 0], [0, 0, 1], [0, 0, ar]))
    Bc = np.array(([0], [0], [br]))

    (A, B) = mpc.util.c2d(Ac, Bc, Delta)  # continuos to discrete

    def simf(x, u):
        return mpc.mtimes(Ac, x) + mpc.mtimes(Bc, u)
    fsim = mpc.getCasadiFunc(simf, [Nx, Nu], ["x", "u"], "fsim")

    def ffunc(x, u):
        return mpc.mtimes(A, x) + mpc.mtimes(B, u)
        #return A*x + B*u
    f = mpc.getCasadiFunc(ffunc, [Nx, Nu], ["x", "u"], "f")

    

    x = np.zeros((Nsim+1,Nx))
    x[0,:] = [0,0,0]
    x0 = x[0,:]
    u = np.zeros((Nsim,Nu))
    pred = []
    upred = []
    avgt = np.array([0])
    main_loop = time()  # return time in sec
    par = np.zeros((Nx+Nu,Nt,Nsim))
    mean_y = 0
    mean_phi = 0
    mean_r = 0
    mean_delta = 0
    mse = 0
    max_x = 0
    max_y = 0
    xz = []
    yz = []
    for t in range(Nsim):
        for k in range(Nt):
            if(t+k > Nsim-1):
                p[k, 0] = b[Nsim-1]  # y_ref
                p[k, 1] = np.arctan2(b[Nsim-1]-b[Nsim-2],
                                    a[Nsim-1]-a[Nsim-2])  # phi_ref
            elif(t + k == 0):
                p[k, 0] = b[k+t]  # y_ref
                p[k, 1] = 0  # phi_ref
            else:
                p[k, 0] = b[k+t]  # y_ref
                p[k, 1] = np.arctan2(b[k+t]-b[k+t-1], a[k+t]-a[k+t-1])  # phi_ref
            if(t+k < 2):
                phi_ref_plus = np.arctan2(b[k+1+t]-b[k+t], a[k+1+t]-a[k+t])
                phi_ref_plus2 = np.arctan2(b[k+2+t]-b[k+1+t], a[k+2+t]-a[k+1+t])
                # r_ref = (phi_ref+ - phi_ref)/Delta
                p[k, 2] = (phi_ref_plus - p[k, 1])/Delta
                p[k, 3] = (((phi_ref_plus2 - 2*phi_ref_plus +
                            p[k, 1])/Delta**2) - ar*p[k, 2])/br
            elif(t+k > Nsim-3):
                # r_ref = (phi_ref - phi_ref-)/Delta
                p[k, 2] = (p[k, 1] - par[1, k, t-1])/Delta
                p[k, 3] = (
                    ((p[k, 1] - 2*par[1, k, t-1] + par[1, k-1, t-1])/Delta**2) - ar*p[k, 2])/br
            else:
                phi_ref_plus = np.arctan2(b[k+1+t]-b[k+t], a[k+1+t]-a[k+t])
                # r_ref = (phi_ref+ - phi_ref-)/2*Delta
                p[k, 2] = (phi_ref_plus - par[1, k, t-1])/(2*Delta)
                #delta_ref = (d/dx(r_ref) - ar*r_ref)/br
                p[k, 3] = (
                    ((phi_ref_plus - 2*p[k, 1] + par[1, k, t-1])/Delta**2) - ar*p[k, 2])/br
            par[:, k, t] = p[k, :]
        
        # #model LTV
        # Ac = np.array(([0, c[t], 0], [0, 0, 1], [0, 0, ar]))
        # #Ac = np.array(([0, c, 0], [0, 0, 1], [0, 0, ar]))
        # Bc = np.array(([0], [0], [br]))

        # (A, B) = mpc.util.c2d(Ac, Bc, Delta)  # continuos to discrete

        # def simf(x, u):
        #     return mpc.mtimes(Ac, x) + mpc.mtimes(Bc, u)
        # fsim = mpc.getCasadiFunc(simf, [Nx, Nu], ["x", "u"], "fsim")

        # def ffunc(x, u):
        #     return mpc.mtimes(A, x) + mpc.mtimes(B, u)
        #     #return A*x + B*u
        # f = mpc.getCasadiFunc(ffunc, [Nx, Nu], ["x", "u"], "f")

        #controlador
        solver = mpc.nmpc(f, N=N, verbosity=0, l=l, x0=x0, lb=lb, ub=ub,
                        p=p, funcargs=funcargs, inferargs=True, uprev=np.array([0]))

        #simulator
        model = mpc.DiscreteSimulator(fsim, Delta, [Nx, Nu], ["x", "u"])

        # Solve nlp.
        t1 = time()
        for k in range(Nt):  # passing parameters along the horizon control
            solver.par["p", k] = par[:, k, t]  # trajectory is the parameter "p"

        solver.solve()  # solving

        # Print stats.
        print("%d: %s" % (t, solver.stats["status"]))
        solver.saveguess()
        solver.fixvar("x", 0, solver.var["x", 1])  # x[k+1] <- x[k]
        u[t, :] = np.array(solver.var["u", 0, :]).flatten()
        
        #predicted trajectories
        pred += [solver.var["x", :, :]]
        upred += [solver.var["u", :, :]]

        # Simulate.
        x[t+1, :] = model.sim(x[t, :], u[t, :])
        x0 = x[t+1, :]  # build vector x
        t2 = time()
        avgt = np.vstack((
            avgt,
            t2-t1
        ))
        if(t == 0):
            xz += [0]
        else:        
            #xz += [xz[t-1] + c[t]*np.cos(x[t,1])*Delta]
            xz += [xz[t-1] + uref*np.cos(x[t,1])*Delta]
        yz += [x[t,0]]
        mean_y = mean_y + ((x[t, 0]-par[0, 0, t])**2)/Nsim
        mean_phi = mean_phi + ((x[t, 1]-par[1, 0, t])**2)/Nsim
        mean_r = mean_r + ((x[t, 2]-par[2, 0, t])**2)/Nsim
        mean_delta = mean_delta + ((u[t]-par[3, 0, t])**2)/Nsim
        traj = np.array([xz,yz])
        traje = np.array([a,b])
        dist = dist + np.linalg.norm(traj[:,t]-traje[:,t])/Nsim
        mse = mse + dist/(t+1)
        # if(max_x < abs(dist)):
        #     max_x = abs(dist)
    main_loop_time = time()
    times = Delta*Nsim*np.linspace(0,1,Nsim+1)
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
    # print('Total time: ', main_loop_time - main_loop)
    # print(mean_y)
    # print(mean_phi)
    # print(mean_r)
    # print(mean_delta)
    # print(mse)    
    # print('J =',mse)

    fig, axs = plt.subplots(3, 2)
    # axs[0, 0].plot(times,x[:,0], label ='actual')
    # axs[0, 0].plot(times[:-1],par[0,0,:], label ='reference')
    # axs[0, 0].set_ylabel("Lateral position")
    # axs[0, 0].set_xlabel("t[s]")
    # axs[0, 0].legend()
    # axs[1, 0].plot(times,x[:,1],label ='actual')
    # axs[1, 0].plot(times[:-1],par[1,0,:],label ='reference')
    # axs[1, 0].set_ylabel("Yaw angle")
    # axs[1, 0].set_xlabel("t[s]")
    # axs[1, 0].legend()
    # axs[2, 0].plot(times,x[:,2], label ='actual')
    # axs[2, 0].plot(times[:-1],par[2,0,:], label ='reference')
    # axs[2, 0].set_ylabel("Yaw rate")
    # axs[2, 0].set_xlabel("t[s]")
    # axs[2, 0].legend()
    # axs[0, 1].step(times[:-1],u, label ='actual')
    # axs[0, 1].plot(times[:-1],par[3,0,:], label ='reference')
    # axs[0, 1].set_ylabel("Steering angle")
    # axs[0, 1].set_xlabel("t[s]")
    # axs[1, 1].legend()
    # axs[1, 1].plot(xz,yz, label = 'actual trajectory')
    # axs[1, 1].plot(a,b, label = 'reference trajectory')
    # axs[1, 1].set_ylabel("y[m]")
    # axs[1, 1].set_xlabel("x[m]")
    # axs[1, 1].legend()
    # axs[2,1].plot(times,avgt, label = '')
    # plt.show()
    axs[0, 0].plot(times,x[:,0], label ='realizado')
    axs[0, 0].plot(times[:-1],par[0,0,:], label ='referencia')
    axs[0, 0].set_ylabel("Posição lateral[m]")
    axs[0, 0].set_xlabel("t[s]")
    axs[0, 0].legend()
    axs[1, 0].plot(times,x[:,1],label ='realizado')
    axs[1, 0].plot(times[:-1],par[1,0,:],label ='referencia')
    axs[1, 0].set_ylabel("Ângulo de guinada[rad]")
    axs[1, 0].set_xlabel("t[s]")
    axs[1, 0].legend()
    axs[2, 0].plot(times,x[:,2], label ='realizado')
    axs[2, 0].plot(times[:-1],par[2,0,:], label ='referencia')
    axs[2, 0].set_ylabel("Velocidade angular[rad/s]")
    axs[2, 0].set_xlabel("t[s]")
    axs[2, 0].legend()
    axs[0, 1].step(times[:-1],u, label ='realizado')
    axs[0, 1].plot(times[:-1],par[3,0,:], label ='referencia')
    axs[0, 1].set_ylabel("Ângulo de esterçamento[rad]")
    axs[0, 1].set_xlabel("t[s]")
    axs[0, 1].legend()
    axs[1, 1].plot(xz,yz, label = 'Trajetória realizada')
    axs[1, 1].plot(a,b, label = 'Trajetória de referência')
    axs[1, 1].set_ylabel("y[m]")
    axs[1, 1].set_xlabel("x[m]")
    axs[1, 1].legend()
    #axs[1,1].plot(np.linspace(0,Nsim+1,Nsim+1),avgt*1000, label = 'Tempo de processamento por iteração')
    #axs[1, 1].set_ylabel("tempo[ms]")
    #axs[1, 1].set_xlabel("iteração")
    #axs[1,1].legend()
    axs[2,1].set_visible(False)
    fig.suptitle('Nt = %d, Q_y = %d, Q_psi = %d, e = %.2f' %(Nt,Q_y,Q_theta,dist))
    # print(mse)
    # print(max_x)
    plt.show()
    # plt.savefig(fname = 'ltv.png', orientation = 'landscape')
    # f = plt.figure(1)
    # plt.plot(xz,x[:-1,0]-par[0,0,:], label = 'Q_phi = %d' %Q_theta)
    # plt.title("Erro lateral na mudança de faixa para diferentes valores de Q_phi")    
    # plt.ylabel("erro lateral [m]")
    # plt.xlabel("deslocamento longitudinal [m]")
    # plt.legend()    

    # g = plt.figure(2)
    # plt.plot(xz,x[:-1,1]-par[1,0,:], label = 'Q_phi = %d' %Q_theta)
    # plt.title("Erro angular na mudança de faixa para diferentes valores de Q_phi")    
    # plt.ylabel("erro angular [rad]")
    # plt.xlabel("deslocamento longitudinal [m]")    
    # plt.legend()
    
    # h = plt.figure(3)
    # plt.plot(xz,(x[:-1,1]-par[1,0,:])*(180/np.pi), label = 'Q_phi = %d' %Q_theta)
    # plt.title("Erro angular na mudança de faixa para diferentes valores de Q_phi")    
    # plt.ylabel("erro angular [graus]")
    # plt.xlabel("deslocamento longitudinal [m]")    
    # plt.legend()
    
    # axs[0].plot(xz,x[:-1,0]-par[0,0,:], label = 'Nt =%d' %Nt)
    # #axs[0].plot(xz,x[:-1,0]-par[0,0,:], label = 'Q_y =%.2f' %Q_y)
    # axs[0].set_ylabel("erro lateral [m]")
    # axs[0].set_xlabel("deslocamento longitudinal [m]")
    # axs[0].legend()
    # axs[1].plot(xz,(x[:-1,1]-par[1,0,:])*(180/np.pi))
    # axs[1].set_ylabel("erro angular [graus]")
    # axs[1].set_xlabel("deslocamento longitudinal [m]")
    # axs[1].legend()
    # axs[2].plot(xz,yz, label = 'e = %.3f, t_p = %.2f ms'%(dist,avg))
    # axs[2].set_ylabel("y[m]")
    # axs[2].set_xlabel("x[m]")
    # axs[2].legend()

    



    #simulate(pred4.T, upred3.T, times, Delta, Nt,
    #              np.array([0, 0, 0, 50, 50, 0]), save=False)

    #fig = mpc.plots.mpcplot(x,u,times, xnames = ["Lateral Positon","Yaw Angle", "Angular velocity"], unames= ["Steering angle"])
    #plt.show()
    #mpc.plots.showandsave(fig,"my_mpctools.pdf")
    # simulate(pred3.T, upred3.T, times, Delta, Nt,
    #              np.array([0, 0, 0, x_target, y_target, theta_target]), save=False)
    #fig = mpc.plots.mpcplot(x,u,times, xnames = ["x Position","y Position", "Angular Displacement"], unames= ["Velocity","Angular Velocity"])

    #mpc.plots.showandsave(fig,"verde_trajectory_tracking.pdf")
# axs[2].plot(a,b, label = 'Trajetória de referência', linestyle = 'dashed')
# axs[2].legend()
# fig.suptitle("Erro para diferentes valores de Nt e Q_y = %d" %Q_y)
# #fig.suptitle("Erro para diferentes valores de Q_y e Nt = %d"%Nt)
# plt.show()
# f.show()
# g.show()
# h.show()
# input()
h= np.hstack((x[1:],u,traj.T,par[:,0,:].T))
g2 = pd.DataFrame(h)
g2.to_csv('dados2.csv',index = False)