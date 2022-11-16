import numpy as np
from scipy.integrate import quad

def reference(x):
    dx = x[1:]-x[:-1]

    # Parameters
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

    # Integrands for current
    def mu_psi1_plus(mu,x):
        return mu*psi1_plus(mu,x)
    def mu_psi2_plus(mu,x):
        return mu*psi2_plus(mu,x)
    def mu_psi3_plus(mu,x):
        return mu*psi3_plus(mu,x)
    def mu_psi1_minus(mu,x):
        return mu*psi1_minus(mu,x)
    def mu_psi2_minus(mu,x):
        return mu*psi2_minus(mu,x)
    def mu_psi3_minus(mu,x):
        return mu*psi3_minus(mu,x)

    # Current
    def J1(x):
        return quad(mu_psi1_plus, 0, 1, args=(x))[0] + quad(mu_psi1_minus, -1, 0, args=(x))[0]
    def J2(x):
        return quad(mu_psi2_plus, 0, 1, args=(x))[0] + quad(mu_psi2_minus, -1, 0, args=(x))[0]
    def J3(x):
        return quad(mu_psi3_plus, 0, 1, args=(x))[0] + quad(mu_psi3_minus, -1, 0, args=(x))[0]

    phi      = np.zeros(60)
    J        = np.zeros(60)
    phi_x = np.zeros(61)
    J_x   = np.zeros(61)

    for i in range(20):
        phi[i]      = quad(phi1,x[i],x[i+1])[0]/dx[i]
        J[i]        = quad(J1,x[i],x[i+1])[0]/dx[i]
        phi_x[i] = phi1(x[i])
        J_x[i]   = J1(x[i])
    for i in range(20,40):
        phi[i]      = quad(phi2,x[i],x[i+1])[0]/dx[i]
        J[i]        = quad(J2,x[i],x[i+1])[0]/dx[i]
        phi_x[i] = phi2(x[i])
        J_x[i]   = J2(x[i])
    for i in range(40,60):
        phi[i]      = quad(phi3,x[i],x[i+1])[0]/dx[i]
        J[i]        = quad(J3,x[i],x[i+1])[0]/dx[i]
        phi_x[i] = phi3(x[i])
        J_x[i]   = J3(x[i])
    phi_x[60] = phi3(x[60])
    J_x[60]   = J3(x[60])

    return phi, phi_x, J, J_x
