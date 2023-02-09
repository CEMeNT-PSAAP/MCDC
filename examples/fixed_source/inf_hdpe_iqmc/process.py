import numpy as np
import matplotlib.pyplot as plt
import h5py



with h5py.File('output.h5', 'r') as f:
    phi   = f['tally/iqmc_flux'][:]
    x     = f['iqmc/grid/x'][:]
    dx    = (x[1]-x[0])
    x_mid = 0.5*(x[:-1]+x[1:])
    f.close()


# =============================================================================
# Plot
# =============================================================================
    
Nx = phi.shape[1]
G  = phi.shape[0]
# Flux - spatial average
plt.figure(dpi=300,figsize=(8,5))
# plt.plot(x_mid,phi_ref,label='Sol')
for i in range(G):
    plt.plot(x_mid,phi[i,:],label='iQMC')
plt.ylabel(r'$\phi(x)$')
plt.xlabel(r'$x$')
plt.grid()

