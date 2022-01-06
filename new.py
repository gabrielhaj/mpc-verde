import casadi as ca
N = 5
# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# Formulate the NLP
e = 0
for k in range(N):
    # New NLP variable for the control
    for a in range(e,e+2):
        Uk = ca.SX.sym('U_' + str(a))    
        w += [Uk]
     
    Uk = ca.vertcat(w[e],w[e+1])
    e = e +2
    print(Uk)
print(w)