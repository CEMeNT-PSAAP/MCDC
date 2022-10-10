import math
import numpy as np
import h5py

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
	Nx = len(x_mid)
	Ny = len(y_mid)

	cf = 5*5

	n    = f['tally/n/mean'][:]
	phi    = f['tally/flux/mean'][:]*4*cf
	phi_sd = f['tally/flux/sdev'][:]*4*cf
	J = f['tally/current/mean'][:]*4*cf
	T = f['runtime'][:]

Jx = J[:,:,0]
Jy = J[:,:,1]

rel_var = phi_sd*phi_sd/phi/phi #MC relative variance
FOM = 1/np.sum(n)/rel_var			#piecewise figure of merit
FOM[n==0]=0

#with np.printoptions(threshold=np.inf):
#	print(J[:,:,0])

def print_var(outfile, var):
			outfile.write(' iy/ix')
			for i in range(Nx):
					outfile.write('%12d' % (i+1))
			outfile.write('\n')
			for j in range(Ny):
					outfile.write('%6d'%(j+1))
					for i in range(Nx):
							outfile.write('%12.4e'%var[i][j])
					outfile.write('\n')

#write file for phi
with open('phi.txt', 'w') as outfile:
    print_var(outfile, phi)

#write file for standard deviation
with open('phi_sd.txt', 'w') as outfile:
    print_var(outfile, phi_sd)

#write file for local Figure of Merit
with open('FOM.txt', 'w') as outfile:
    print_var(outfile, FOM)

#write file for n
with open('n.txt', 'w') as outfile:
    print_var(outfile, n)

#write file for Jx
with open('Jx.txt', 'w') as outfile:
    print_var(outfile, Jx)

#write file for Jy
with open('Jy.txt', 'w') as outfile:
    print_var(outfile, Jy)
