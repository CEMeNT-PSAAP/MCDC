import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

output = sys.argv[1]

# =============================================================================
# Reference solution
# =============================================================================

# Load material data
with np.load('SHEM-361.npz') as data:
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    nuSigmaF = data['nuSigmaF']
    G      = data['G']
    E      = data['E']
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

# Buckling and leakage XS to make the problem subcritical
R      = 10.0 # Sub
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaT += SigmaL

A = np.diag(SigmaT) - SigmaS - nuSigmaF
Q = np.ones(G)/G

phi_ref = np.linalg.solve(A,Q)*E_mid/dE

# =============================================================================
# Plot results
# =============================================================================

# Get results
with h5py.File(output, 'r') as f:
    phi      = f['tally/flux/mean'][:]/dE*E_mid
    phi_sd   = f['tally/flux/sdev'][:]/dE*E_mid

# Flux
plt.step(E_mid,phi,'-b',label="MC",where='mid')
plt.fill_between(E_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b',step='mid')
plt.step(E_mid,phi_ref,'--r',label='analytical',where='mid')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()
