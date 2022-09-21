import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import os
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
	t     = f['tally/grid/t'][:]
	dt    = t[1:] - t[:-1]
	K     = len(dt)

	phi    = f['tally/flux/mean'][:]*4
	phi_sd = f['tally/flux/sdev'][:]*4
	#N = f['tally/flux/N'][:]
	T = f['runtime'][:]
	#N_hist = f['N_histories'][:]

#with np.load('wwHybrid.npz','r') as f:
#    phi_t = f['phi'][:]

for k in range(K):
	phi[k] /= dt[k]
	phi_sd[k] /= dt[k]
	#N[k]=N[k].transpose()
	phi[k] = phi[k].transpose()
	phi_sd[k] = phi_sd[k].transpose()
	#phi_t = phi_t.transpose()

rel_var = phi_sd*phi_sd/phi/phi

fig = plt.figure(figsize=(4,3))
ax = fig.gca(projection='3d')
#ax.plot_surface(X,Y,np.log10(1/T/rel_var),color='w',edgecolors='k', lw=0.1)
#ax.plot_surface(X,Y,np.log10(abs(phi_t-phi)/phi_t),color='w',edgecolors='k', lw=0.1)
#ax.plot_surface(X,Y,np.log10(phi[-1]),color='w',edgecolors='k', lw=0.1)
#ax.plot_surface(X,Y,np.log10(N),color='w',edgecolors='k', lw=0.1)
#ax.azim = 50
#ax.elev = 24
#ax.dist = 9
#ax.zaxis.set_rotate_label(False)
#ax.set_zlabel(r'Log10 of Figure of Merit', rotation=90)
#ax.set_zlabel(r'Log10 of Relative Error in Weight Window', rotation=90)
#ax.set_zlabel(r'Log10 of Average Number of Monte Carlo Particle Tracks', rotation=90)
#plt.show()
filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(phi[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-10,1)
	#ax.set_zlim(-1,4)
	#plt.draw()
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	#plt.pause(0.1)
	ax.cla()

with imageio.get_writer('out.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

