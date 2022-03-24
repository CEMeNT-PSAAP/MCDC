import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

k_exact = 1.28657
phi_exact = np.array([1, 1.417721, 1.698988, 1.903163, 2.03435, 2.092069, 2.075541, 1.984535, 1.818753, 1.574144, 1.199995, 0.9532296, 0.7980474, 0.6788441, 0.5823852, 0.5020479, 0.4337639, 0.3747058, 0.3226636, 0.2755115, 0.228371])
tmp = 0.5*(phi_exact[1:] + phi_exact[:-1])
phi_exact /= np.sum(tmp)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    x   = f['tally/grid/x'][:]
    dx  = x[1:] - x[:-1]
    phi = f['tally/flux/mean'][:]
    k   = f['keff'][:]
    x_mid = 0.5*(x[:-1]+x[1:])
    
# Get average
N_passive = 10
N_active = 30
phi_avg = np.zeros_like(phi[0])
phi_sd  = np.zeros_like(phi[0])
k_avg = 0.0
k_sd  = 0.0
for i in range(N_passive,len(phi)): 
    phi_avg += phi[i][:]/dx
    phi_sd  += np.square(phi[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
phi_avg /= N_active    
phi_sd  = np.sqrt((phi_sd/N_active - np.square(phi_avg))/(N_active-1))
norm = np.sum(phi_avg)
phi_avg /= norm
phi_sd /= norm
k_avg /= N_active    
k_sd  = np.sqrt((k_sd/N_active - np.square(k_avg))/(N_active-1))
    
# Plot
N_iter = len(k)
plt.plot(np.arange(2,N_iter+1),k[1:],'-b',label='MC')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_exact,'--r',label='analytical')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_avg,':g',label='MC-avg')
plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(k_avg-k_sd),np.ones(N_iter-1)*(k_avg+k_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$k$')
plt.grid()
plt.legend()
plt.show()

plt.plot(x_mid,phi_avg,'-ob',fillstyle='none',label="MC")
plt.fill_between(x_mid,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
plt.plot(x,phi_exact,'--xr',label='analytical')
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')
plt.grid()
plt.legend()
plt.show()
