import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi    = f['tally/flux/mean'][:]
    phi_sd = f['tally/flux/sdev'][:]
    k      = f['keff'][:]
    rg     = f['gyration_radius'][:]
    x      = f['tally/grid/x'][:]
    y      = f['tally/grid/y'][:]

dx  = x[1] - x[0]
x_mid = 0.5*(x[1:] + x[:-1])
y_mid = 0.5*(y[1:] + y[:-1])
Y,X = np.meshgrid(x_mid, y_mid)

# Get average and stdv over the active iterations
N_passive = 10
N_active = 10
phi_avg = np.zeros_like(phi[0])
phi_sd  = np.zeros_like(phi[0])
k_avg = 0.0
k_sd  = 0.0
for i in range(N_passive,len(phi)): 
    phi_avg += phi[i][:]
    phi_sd  += np.square(phi[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
phi_avg /= N_active
phi_sd  = np.sqrt((phi_sd/N_active - np.square(phi_avg))/(N_active-1))
norm = np.sum(phi_avg)
phi_avg = phi_avg/norm/dx**2
phi_sd  = phi_sd/norm/dx**2
k_avg /= N_active    
k_sd  = np.sqrt((k_sd/N_active - np.square(k_avg))/(N_active-1))

#phi_avg = np.sum(phi_avg, axis=0)
#phi_sd  = np.sqrt(np.sum(np.square(phi_sd), axis=0))

phi_fast    = np.zeros_like(phi_avg[0])
phi_thermal = np.zeros_like(phi_avg[0])

for i in range(5):
    phi_fast += phi_avg[i]
for i in range(5,7):
    phi_thermal += phi_avg[i]

# Plot
N_iter = len(k)
p1, = plt.plot(np.arange(1,N_iter+1),k,'-b',label='MC')
p2, = plt.plot(np.arange(1,N_iter+1),np.ones(N_iter)*k_avg,':r',label='MC-avg')
plt.fill_between(np.arange(1,N_iter+1),np.ones(N_iter)*(k_avg-k_sd),np.ones(N_iter)*(k_avg+k_sd),alpha=0.2,color='r')
plt.xlabel('Iteration #')
plt.ylabel(r'$k$')
plt.grid()
ax2 = plt.gca().twinx()
p3, = ax2.plot(np.arange(1,N_iter+1),rg,'g--',label='GyRad')
plt.ylabel(r'Gyration radius [cm]')
lines = [p1, p2, p3]
plt.legend(lines, [l.get_label() for l in lines])
plt.show()

plt.pcolormesh(X,Y,phi_fast,shading='nearest')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [cm]')
plt.ylabel(r'$y$ [cm]')
plt.title(r'Fast neutron flux')
plt.show()

plt.pcolormesh(X,Y,phi_thermal,shading='nearest')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [cm]')
plt.ylabel(r'$y$ [cm]')
plt.title(r'Thermal neutron flux')
plt.show()
