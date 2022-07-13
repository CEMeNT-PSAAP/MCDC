import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

output = sys.argv[1]

# =============================================================================
# Reference solution
# =============================================================================

x  = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.6, 
              1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
dx = x[1:] - x[:-1]
k_exact = 1.28657
phi_exact = np.array([1, 1.417721, 1.698988, 1.903163, 2.03435, 2.092069, 
                      2.075541, 1.984535, 1.818753, 1.574144, 1.199995, 
                      0.9532296, 0.7980474, 0.6788441, 0.5823852, 0.5020479, 
                      0.4337639, 0.3747058, 0.3226636, 0.2755115, 0.228371])
tmp = 0.5*(phi_exact[1:] + phi_exact[:-1])
phi_exact /= np.sum(tmp*dx)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File(output, 'r') as f:
    x       = f['tally/grid/x'][:]
    dx      = x[1:] - x[:-1]
    phi_avg = f['tally/flux-x/mean'][:]
    phi_sd  = f['tally/flux-x/sdev'][:]
    k       = f['k_cycle'][:]
    k_avg   = f['k_mean'][()]
    k_sd    = f['k_sdev'][()]
    rg      = f['gyration_radius'][:]

tmp = 0.5*(phi_avg[1:] + phi_avg[:-1])
norm = np.sum(tmp*dx)
phi_avg /= norm
phi_sd /= norm
    
# Plot
N_iter = len(k)
p1, = plt.plot(np.arange(1,N_iter+1),k,'-b',label='MC')
p2, = plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_exact,'--r',label='analytical')
p3, = plt.plot(np.arange(1,N_iter+1),np.ones(N_iter)*k_avg,':g',label='MC-avg')
plt.fill_between(np.arange(1,N_iter+1),np.ones(N_iter)*(k_avg-k_sd),np.ones(N_iter)*(k_avg+k_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$k$')
plt.grid()
ax2 = plt.gca().twinx()
p4, = ax2.plot(np.arange(1,N_iter+1),rg,'-.m',label='GyRad')
plt.ylabel(r'Gyration radius [cm]')
lines = [p1, p2, p3, p4]
plt.legend(lines, [l.get_label() for l in lines])
plt.show()

plt.plot(x,phi_avg,'-ob',fillstyle='none',label="MC")
plt.fill_between(x,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
plt.plot(x,phi_exact,'--xr',label='analytical')
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')
plt.grid()
plt.legend()
plt.show()
