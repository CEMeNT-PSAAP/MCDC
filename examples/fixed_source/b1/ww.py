import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================

# Results
v=1.383
pi=math.acos(-1)

with h5py.File('output.h5', 'r') as f:
	x     = f['tally/grid/x'][:]
	x_mid = 0.5*(x[:-1]+x[1:])
	y     = f['tally/grid/y'][:]
	y_mid = 0.5*(y[:-1]+y[1:])
	X,Y = np.meshgrid(x_mid,y_mid)

	cf = 1.0

	phi    = f['tally/flux/mean'][:]*100*cf
	phi_sd = f['tally/flux/sdev'][:]*100*cf
	T = f['runtime'][:]

ww=phi/np.max(phi)
np.savez("ww.npz",phi=ww)

ww = ww.transpose()

#write file for standard deviation
with open('ww.txt', 'w') as outfile:
	np.savetxt(outfile, ww, fmt='%-10.4e')
