import xxlimited
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    x       = f['tally/grid/x'][:]
    dx      = x[1:] - x[:-1]
    phi_avg = f['tally/flux/mean'][:]
    phi_sd  = f['tally/flux/sdev'][:]

tmp = 0.5*(phi_avg[1:] + phi_avg[:-1])
norm = np.sum(tmp*dx)
phi_avg /= norm
phi_sd /= norm
    
xx = np.linspace(0,1,99)

plt.figure(1)
plt.plot(xx, phi_avg[0,:], label='0')
plt.plot(xx, phi_avg[1,:], label='1')
plt.plot(xx, phi_avg[2,:], label='2')
#plt.plot(xx, phi_avg[3,:], label='3')
plt.show()
