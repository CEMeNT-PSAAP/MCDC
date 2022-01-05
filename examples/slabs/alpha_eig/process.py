import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

k_exact = 1.
alpha_exact = 0.14247481
phi_exact = np.array([1,1.4573313,1.7677868,1.994216,2.1400088,2.2040045,2.1851026,2.0831691,1.8983807,1.6272924,1.2166341,0.94899798,0.78264174,0.65635543,0.55539366,0.47233885,0.4026271,0.34309503,0.29129485,0.24493022,0.19900451])
phi_exact /= np.sum(phi_exact)

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi = f['tally/flux-face/mean'][:]
    x = f['tally/spatial_grid'][:]
    k   = f['keff'][:]
    alpha   = f['alpha_eff'][:]
    
# Normalize eigenvector
for i in range(len(phi)): 
    phi[i][:] /= np.sum(phi[i])

# Get average (50 passive cycles)
phi_avg = np.zeros_like(phi[0])
phi_sd  = np.zeros_like(phi[0])
k_avg = 0.0
k_sd  = 0.0
alpha_avg = 0.0
alpha_sd  = 0.0
for i in range(10,len(phi)): 
    phi_avg += phi[i][:]
    phi_sd  += np.square(phi[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
    alpha_avg += alpha[i]
    alpha_sd  += alpha[i]**2
phi_avg /= 100    
phi_sd  = np.sqrt((phi_sd/100 - np.square(phi_avg))/(100-1))
k_avg /= 100    
k_sd  = np.sqrt((k_sd/100 - np.square(k_avg))/(100-1))
alpha_avg /= 100    
alpha_sd  = np.sqrt((alpha_sd/100 - np.square(alpha_avg))/(100-1))
phi_avg = phi_avg
phi_sd  = phi_sd
    
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

# Plot
N_iter = len(alpha)
plt.plot(np.arange(2,N_iter+1),alpha[1:],'-b',label='MC')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*alpha_exact,'--r',label='analytical')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*alpha_avg,':g',label='MC-avg')
plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(alpha_avg-alpha_sd),np.ones(N_iter-1)*(alpha_avg+alpha_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$\alpha$')
plt.grid()
plt.legend()
plt.show()

plt.plot(x,phi_avg,'-b',label="MC")
plt.fill_between(x,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
plt.plot(x,phi_exact,'--r',label='analytical')
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')
plt.grid()
plt.legend()
plt.show()
