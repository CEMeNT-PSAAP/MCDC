import numpy as np
from scipy.integrate import quad

def phi(N):
    SigmaT3 = 1.0
    SigmaT1 = 1.5
    SigmaT2 = 2.0
    q1 = 1.0/6.0/2
    q2 = 1.0/6.0/2
    q3 = 1.0/6.0/2
    x1 = 2.0
    x2 = 4.0
    x3 = 6.0
    tau1 = SigmaT1*x1
    tau2 = SigmaT2*(x2-x1)
    tau3 = SigmaT3*(x3-x2)

    # Angular flux
    def psi1_plus(mu,x):
        return q1/SigmaT1*(1.0 - np.exp(-SigmaT1*x/mu))
    def psi2_plus(mu,x):
        return (psi1_plus(mu,x1) - q2/SigmaT2)*np.exp(-SigmaT2*(x-x1)/mu) + q2/SigmaT2
    def psi3_plus(mu,x):
        return (psi2_plus(mu,x2) - q3/SigmaT3)*np.exp(-SigmaT3*(x-x2)/mu) + q3/SigmaT3
    def psi1_minus(mu,x):
        return (psi2_minus(mu,x1) - q1/SigmaT1)*np.exp(-SigmaT1*(x1-x)/np.abs(mu)) + q1/SigmaT1
    def psi2_minus(mu,x):
        return (psi3_minus(mu,x2) - q2/SigmaT2)*np.exp(-SigmaT2*(x2-x)/np.abs(mu)) + q2/SigmaT2
    def psi3_minus(mu,x):
        return q3/SigmaT3*(1.0 - np.exp(-SigmaT3*(x3-x)/np.abs(mu)))

    # Flux
    def phi1(x):
        return (quad(psi1_plus, 0, 1, args=(x))[0] + quad(psi1_minus, -1, 0, args=(x))[0])
    def phi2(x):
        return (quad(psi2_plus, 0, 1, args=(x))[0] + quad(psi2_minus, -1, 0, args=(x))[0])
    def phi3(x):
        return (quad(psi3_plus, 0, 1, args=(x))[0] + quad(psi3_minus, -1, 0, args=(x))[0])
    
    phi_ref      = np.zeros(N)

    part = int(N/3)
    x = np.linspace(0.0,6.0,N+1)
    dx = x[1]-x[0]

    for i in range(part):
        phi_ref[i]      = quad(phi1,x[i],x[i+1])[0]/dx
    for i in range(part,2*part):
        phi_ref[i]      = quad(phi2,x[i],x[i+1])[0]/dx
    for i in range(2*part,3*part):
        phi_ref[i]      = quad(phi3,x[i],x[i+1])[0]/dx

    return phi_ref
