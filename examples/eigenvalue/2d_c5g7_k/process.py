import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    P    = f['tally/fission/mean'][:]
    P_sd = f['tally/fission/sdev'][:]
    k    = f['keff'][:]
    x    = f['tally/grid/x'][:]
    y    = f['tally/grid/y'][:]

dx  = x[1] - x[0]
x_mid = 0.5*(x[1:] + x[:-1])
y_mid = 0.5*(y[1:] + y[:-1])
Y,X = np.meshgrid(x_mid, y_mid)

# Get average and stdv over the active iterations
N_passive = 10
N_active = 10
P_avg = np.zeros_like(P[0])
P_sd  = np.zeros_like(P[0])
k_avg = 0.0
k_sd  = 0.0
for i in range(N_passive,len(P)): 
    P_avg += P[i][:]
    P_sd  += np.square(P[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
P_avg /= N_active
P_sd  = np.sqrt((P_sd/N_active - np.square(P_avg))/(N_active-1))
norm = np.sum(P_avg)
P_avg = P_avg/norm/dx**2
P_sd  = P_sd/norm/dx**2
k_avg /= N_active    
k_sd  = np.sqrt((k_sd/N_active - np.square(k_avg))/(N_active-1))

P_avg = np.sum(P_avg, axis=0)
P_sd  = np.sqrt(np.sum(np.square(P_avg), axis=0))

# Plot
N_iter = len(k)
plt.plot(np.arange(2,N_iter+1),k[1:],'-b',label='MC')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_avg,':g',label='MC-avg')
plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(k_avg-k_sd),np.ones(N_iter-1)*(k_avg+k_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$k$')
plt.grid()
plt.legend()
plt.show()

plt.pcolormesh(X,Y,P_avg,shading='nearest')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [cm]')
plt.ylabel(r'$y$ [cm]')
plt.show()
