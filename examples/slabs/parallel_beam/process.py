import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/grid/x'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# XS
SigmaT1 = 1.0
SigmaT2 = 1.5
SigmaT3 = 2.0

# Spatial average flux
phi_ref = -(1-np.exp(SigmaT2*dx))*np.exp(-SigmaT2*x[1:21])/SigmaT2/dx
phi_ref = np.append(phi_ref, np.exp(-SigmaT2*2.0)*-(1-np.exp(SigmaT3*dx))*np.exp(-SigmaT3*x[1:21])/SigmaT3/dx)
phi_ref = np.append(phi_ref, np.exp(-(SigmaT2+SigmaT3)*2.0)*-(1-np.exp(SigmaT1*dx))*np.exp(-SigmaT1*x[1:21])/SigmaT1/dx)

# =============================================================================
# Plot results
# =============================================================================

# Neutron flux results
with h5py.File('output.h5', 'r') as f:
    phi         = f['tally/flux/mean'][:]/dx
    phi_sd      = f['tally/flux/sdev'][:]/dx

# Plots
plt.plot(x_mid,phi,'-b',label="MC")
plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(x_mid,phi_ref,'--r',label="ref.")
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.ylim([1E-4,1E0])
plt.grid()
plt.yscale('log')
plt.legend()
plt.title(r'$\bar{\phi}_i$')
plt.show()
