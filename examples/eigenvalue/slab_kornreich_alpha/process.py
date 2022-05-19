import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

x  = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.6, 
              1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
dx = x[1:] - x[:-1]
k_exact = 1.0
alpha_exact = 0.14247481
phi_exact = np.array([1,1.4573313,1.7677868,1.994216,2.1400088,2.2040045,2.1851026,2.0831691,1.8983807,1.6272924,1.2166341,0.94899798,0.78264174,0.65635543,0.55539366,0.47233885,0.4026271,0.34309503,0.29129485,0.24493022,0.19900451])
tmp = 0.5*(phi_exact[1:] + phi_exact[:-1])
phi_exact /= np.sum(tmp*dx)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    x   = f['tally/grid/x'][:]
    dx  = x[1:] - x[:-1]
    phi = f['tally/flux/mean'][:]
    k   = f['keff'][:]
    alpha = f['alpha_eff'][:]
    x_mid = 0.5*(x[:-1]+x[1:])
x = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.6, 
              1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
    
# Get average
N_passive = 10
N_active = 30
phi_avg = np.zeros_like(phi[0])
phi_sd  = np.zeros_like(phi[0])
k_avg = 0.0
k_sd  = 0.0
alpha_avg = 0.0
alpha_sd  = 0.0
for i in range(N_passive,len(phi)): 
    phi_avg += phi[i][:]
    phi_sd  += np.square(phi[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
    alpha_avg += alpha[i]
    alpha_sd  += alpha[i]**2
phi_avg /= N_active    
phi_sd  = np.sqrt((phi_sd/N_active - np.square(phi_avg))/(N_active-1))
norm = np.sum(phi_avg)
phi_avg /= norm
phi_sd /= norm
k_avg /= N_active    
k_sd  = np.sqrt((k_sd/N_active - np.square(k_avg))/(N_active-1))
alpha_avg /= N_active    
alpha_sd  = np.sqrt((alpha_sd/N_active - np.square(alpha_avg))/(N_active-1))
    
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

plt.plot(np.arange(2,N_iter+1),alpha[1:],'-b',label='MC')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*alpha_exact,'--r',label='analytical')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*alpha_avg,':g',label='MC-avg')
plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(alpha_avg-alpha_sd),np.ones(N_iter-1)*(alpha_avg+alpha_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$\alpha$')
plt.grid()
plt.legend()
plt.show()

phi_avg /= dx
phi_sd  /= dx

plt.plot(x_mid,phi_avg,'-ob',fillstyle='none',label="MC")
plt.fill_between(x_mid,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
plt.plot(x,phi_exact,'--xr',label='analytical')
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')
plt.grid()
plt.legend()
plt.show()
