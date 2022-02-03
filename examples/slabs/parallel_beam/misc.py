import numpy as np

def phi(N):
    SigmaT1 = 1.0
    SigmaT2 = 1.5
    SigmaT3 = 2.0
    x  = np.linspace(0.0,6.0,N+1)
    dx = 6.0/N

    part = int(N/3+1)
    phi_ref = -(1-np.exp(SigmaT2*dx))*np.exp(-SigmaT2*x[1:part])/SigmaT2/dx
    phi_ref = np.append(phi_ref, np.exp(-SigmaT2*2.0)*-(1-np.exp(SigmaT3*dx))*np.exp(-SigmaT3*x[1:part])/SigmaT3/dx)
    phi_ref = np.append(phi_ref, np.exp(-(SigmaT2+SigmaT3)*2.0)*-(1-np.exp(SigmaT1*dx))*np.exp(-SigmaT1*x[1:part])/SigmaT1/dx)

    return phi_ref
