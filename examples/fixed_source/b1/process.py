import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
	x     = f['tally/grid/x'][:]
	x_mid = 0.5*(x[:-1]+x[1:])
	y     = f['tally/grid/y'][:]
	y_mid = 0.5*(y[:-1]+y[1:])
	X,Y = np.meshgrid(x_mid,y_mid)

	cf = 1.0

	n    = f['tally/n/mean'][:]
	phi    = f['tally/flux/mean'][:]*100*cf
	phi_sd = f['tally/flux/sdev'][:]*100*cf
	T = f['runtime'][:]

n = n.transpose()
phi = phi.transpose()
phi_sd = phi_sd.transpose()

rel_var = phi_sd*phi_sd/phi/phi #MC relative variance
FOM = 1/T/rel_var			#piecewise figure of merit

#write file for phi
with open('phi.txt', 'w') as outfile:
	np.savetxt(outfile, phi, fmt='%-10.4e')

#write file for standard deviation
with open('phi_sd.txt', 'w') as outfile:
	np.savetxt(outfile, phi_sd, fmt='%-10.4e')

#write file for local Figure of Merit
with open('FOM.txt', 'w') as outfile:
	np.savetxt(outfile, FOM, fmt='%-10.4e')

#write file for n
with open('n.txt', 'w') as outfile:
	np.savetxt(outfile, n, fmt='%-10.4e')
