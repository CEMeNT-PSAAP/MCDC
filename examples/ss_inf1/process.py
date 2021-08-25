import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('XS.npz') as data:
    speeds = data['v']        # cm/s
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu     = data['nu']
    E      = data['E']        # eV
G     = len(speeds)
E_mid = 0.5*(E[:-1] + E[1:])
dE    = E[1:] - E[:-1]

# Augment with uniform leakage XS
SigmaL  = 0.24 # /cm
SigmaT += SigmaL

# Analytical solution
nuSigmaF = SigmaF.dot(np.diag(nu))
A = np.diag(SigmaT) - SigmaS - nuSigmaF
b = np.zeros(G)
b[-1] = 1.0
phi_ref = np.linalg.solve(A,b)
phi_ref = np.divide(phi_ref,dE)
phi_ref = np.multiply(phi_ref,E_mid)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi    = f['tally/flux/mean'][:]/dE*E_mid
    phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              

# Plot
plt.plot(E_mid,phi,'-b',label="MC")
plt.fill_between(E_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(E_mid,phi_ref,'--r',label='ref.')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()
