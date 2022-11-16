import matplotlib.pyplot as plt
import h5py
import sys

from reference import reference


# Load results
output = sys.argv[1]
with h5py.File(output, 'r') as f:
    x        = f['tally/grid/x'][:]
    dx       = x[1:]-x[:-1]
    x_mid    = 0.5*(x[:-1]+x[1:])
    phi      = f['tally/flux/mean'][:]/dx
    phi_sd   = f['tally/flux/sdev'][:]/dx
    phi_x    = f['tally/flux-x/mean'][:]
    phi_x_sd = f['tally/flux-x/sdev'][:]
    J        = f['tally/current/mean'][:,0]/dx
    J_sd     = f['tally/current/sdev'][:,0]/dx
    J_x      = f['tally/current-x/mean'][:,0]
    J_x_sd   = f['tally/current-x/sdev'][:,0]

# Reference solution 
phi_ref, phi_x_ref, J_ref, J_x_ref = reference(x)

# Flux - spatial average
plt.plot(x_mid,phi,'-b',label="MC")
plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(x_mid,phi_ref,'--r',label="ref.")
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.ylim([0.06,0.16])
plt.grid()
plt.legend()
plt.title(r'$\bar{\phi}_i$')
plt.show()

# Flux - spatial grid
plt.plot(x,phi_x,'-b',label="MC")
plt.fill_between(x,phi_x-phi_x_sd,phi_x+phi_x_sd,alpha=0.2,color='b')
plt.plot(x,phi_x_ref,'--r',label="ref.")
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.ylim([0.06,0.16])
plt.grid()
plt.legend()
plt.title(r'$\phi(x)$')
plt.show()

# Current - spatial average
plt.plot(x_mid,J,'-b',label="MC")
plt.fill_between(x_mid,J-J_sd,J+J_sd,alpha=0.2,color='b')
plt.plot(x_mid,J_ref,'--r',label="ref.")
plt.xlabel(r'$x$, cm')
plt.ylabel('Current')
plt.ylim([-0.03,0.045])
plt.grid()
plt.legend()
plt.title(r'$\bar{J}_i$')
plt.show()

# Current - spatial grid
plt.plot(x,J_x,'-b',label="MC")
plt.fill_between(x,J_x-J_x_sd,J_x+J_x_sd,alpha=0.2,color='b')
plt.plot(x,J_x_ref,'--r',label="ref.")
plt.xlabel(r'$x$, cm')
plt.ylabel('Current')
plt.ylim([-0.03,0.045])
plt.grid()
plt.legend()
plt.title(r'$J(x)$')
plt.show()
