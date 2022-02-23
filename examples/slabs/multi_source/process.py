import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    x     = f['tally/grid/x'][:]
    x_mid = 0.5*(x[:-1]+x[1:])
    phi    = f['tally/flux/mean'][:]
    phi_sd = f['tally/flux/sdev'][:]
    J      = f['tally/current/mean'][:,0]
    J_sd   = f['tally/current/sdev'][:,0]

# Flux
plt.plot(x_mid,phi,'-b',label="MC")
plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.grid()
plt.legend()
plt.title(r'$\bar{\phi}_i$')
plt.show()

# Current
plt.plot(x_mid,J,'-b',label="MC")
plt.fill_between(x_mid,J-J_sd,J+J_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Current')
plt.grid()
plt.legend()
plt.title(r'$\bar{J}_i$')
plt.show()
