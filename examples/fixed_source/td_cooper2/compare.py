import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio

# =============================================================================
# Process or Plot results
# =============================================================================
v=1.383
pi=math.acos(-1)
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
	T = f['runtime'][:]


with h5py.File('outputIC.h5', 'r') as f:
	n2    = f['tally/n/mean'][:]
	phi2    = f['tally/flux/mean'][:]*4*cf
	phi_sd2 = f['tally/flux/sdev'][:]*4*cf
	T2 = f['runtime'][:]


Nx = len(x_mid)
Ny = len(y_mid)



for k in range(K):
	phi[k] /= dt[k] 			#Scalar Flux
	phi_sd[k] /= dt[k] 		#MC standard deviation
	n[k] = n[k].transpose()
	phi[k] = phi[k].transpose()
	phi_sd[k] = phi_sd[k].transpose()
	phi2[k] /= dt[k] 			#Scalar Flux
	phi_sd2[k] /= dt[k] 		#MC standard deviation
	n2[k] = n2[k].transpose()
	phi2[k] = phi2[k].transpose()
	phi_sd2[k] = phi_sd2[k].transpose()

	

rel_var = phi_sd*phi_sd/phi/phi #MC relative variance
FOM = 1/np.sum(n)/rel_var			#piecewise figure of merit
FOM[n==0]=0
rel_var2 = phi_sd2*phi_sd2/phi2/phi2 #MC relative variance
FOM2 = 1/np.sum(n2)/rel_var2			#piecewise figure of merit
FOM2[n2==0]=0

phim = np.maximum(phi,phi2)
rd_phi=(phi-phi2)/phim
rn=n/n2
rn_normed=np.zeros_like(rn)
for k in range(K):
	rn_normed[k]=n[k]/np.sum(n[k])*np.sum(n2[k])/n2[k]

rvar=phi_sd*phi_sd/phi_sd2/phi_sd2
rFOM=FOM/FOM2


def print_var(outfile, var):
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        outfile.write('      ')
        outfile.write('        ix')
        for i in range(Nx):
            outfile.write('%12d' % (i+1))
        outfile.write('\n')
        outfile.write('    iy')
        outfile.write('       y/x')
        for i in range(Nx):
            outfile.write('%12.2f'%x_mid[i])
        outfile.write('\n')
        for j in range(Ny):
            outfile.write('%6d'%(j+1))
            outfile.write('%10.2f'%y_mid[j])
            for i in range(Nx):
                outfile.write('%12.4e'%var[k][j][i])
            outfile.write('\n')


#write file for phi
with open('rd_phi.txt', 'w') as outfile:
    print_var(outfile, rd_phi)

#write file for n
with open('rel_n.txt', 'w') as outfile:
    print_var(outfile, rn)

#write file for n normed
with open('rel_n_normed.txt', 'w') as outfile:
    print_var(outfile, rn_normed)

#write file for ratio of variance
with open('rvar.txt', 'w') as outfile:
    print_var(outfile, rvar)

#write file for ratio of FOM
with open('rFOM.txt', 'w') as outfile:
    print_var(outfile, rFOM)


fig = plt.figure(figsize=(16,9))
ax = fig.gca(projection='3d')
filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(rvar[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-2,2)
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	ax.cla()

with imageio.get_writer('rvar.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


for filename in set(filenames):
    os.remove(filename)

filenames = []
for k in range(K):
	ax.plot_surface(X,Y,np.log10(rFOM[k]),color='w',edgecolors='k', lw=0.1)
	ax.set_facecolor('w')
	ax.set_xlabel(r'$x$ [cm]')
	ax.set_ylabel(r'$y$ [cm]')
	ax.set_zlim(-2,2)
	filename = f'{k}.png'
	filenames.append(filename)
	plt.savefig(filename)
	ax.cla()

with imageio.get_writer('rFOM.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

