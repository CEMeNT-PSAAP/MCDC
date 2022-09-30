import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
import math

pi = math.acos(-1)
v = 1.383
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

	cf = 5*5*2+pi*4E-10*20*20/v

	n    = f['tally/n/mean'][:]
	phi    = f['tally/flux/mean'][:]*4*cf
	phi_sd = f['tally/flux/sdev'][:]*4*cf
	nt    = f['tally/n-t/mean'][:]
	phit    = f['tally/flux-t/mean'][:]*4*cf
	phit_sd = f['tally/flux-t/sdev'][:]*4*cf
	T = f['runtime'][:]

#with np.load('ww.npz') as data:
#	ww = data['phi']

#print(ww[0][5:15,1:10])

for k in range(K+1):
	nt[k] = nt[k].transpose()
	phit[k] = phit[k].transpose()
	phit_sd[k] = phit_sd[k].transpose()

alpha = (np.log10(phit[1:])-np.log10(phit[:-1]))

for k in range(K):
	phi[k] /= dt[k] 			#Scalar Flux
	phi_sd[k] /= dt[k] 		#MC standard deviation
	alpha[k] /= dt[k] 	
	n[k] = n[k].transpose()
	phi[k] = phi[k].transpose()
	phi_sd[k] = phi_sd[k].transpose()

	

rel_var = phi_sd*phi_sd/phi/phi #MC relative variance
FOM = 1/T/rel_var			#piecewise figure of merit

fig = plt.figure(figsize=(16,9))
ax = fig.gca(projection='3d')
filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(phi[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-10,1)
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	ax.cla()

with imageio.get_writer('phi.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(n[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-6,0)
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	ax.cla()

with imageio.get_writer('n.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(FOM[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-6,0)
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	ax.cla()

with imageio.get_writer('FOM.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)
