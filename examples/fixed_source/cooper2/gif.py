import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import imageio

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
	x     = f['tally/grid/x'][:]
	x_mid = 0.5*(x[:-1]+x[1:])
	y     = f['tally/grid/y'][:]
	y_mid = 0.5*(y[:-1]+y[1:])
	X,Y = np.meshgrid(x_mid,y_mid)

	n    = f['tally/n/mean'][:]
	phi    = f['tally/flux/mean'][:]*4
	phi_sd = f['tally/flux/sdev'][:]*4
	T = f['runtime'][:]

n=n.transpose()
phi = phi.transpose()
phi_sd = phi_sd.transpose()

rel_var = phi_sd*phi_sd/phi/phi
FOM = 1/np.sum(n)/rel_var			#piecewise figure of merit
FOM[n==0]=0

fig = plt.figure(figsize=(16,9))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,np.log10(phi),color='w',edgecolors='k', lw=0.1)
ax.set_facecolor('w')
ax.set_xlabel(r'$x$ [cm]')
ax.set_ylabel(r'$y$ [cm]')
ax.set_zlim(-8,1)
plt.savefig('phi.png')
ax.cla()

ax.plot_surface(X,Y,np.log10(n),color='w',edgecolors='k', lw=0.1)
ax.set_facecolor('w')
ax.set_xlabel(r'$x$ [cm]')
ax.set_ylabel(r'$y$ [cm]')
ax.set_zlim(-3,3)
plt.savefig('n.png')
ax.cla()

ax.plot_surface(X,Y,np.log10(FOM),color='w',edgecolors='k', lw=0.1)
ax.set_facecolor('w')
ax.set_xlabel(r'$x$ [cm]')
ax.set_ylabel(r'$y$ [cm]')
ax.set_zlim(-3,3)
plt.savefig('FOM.png')
ax.cla()
