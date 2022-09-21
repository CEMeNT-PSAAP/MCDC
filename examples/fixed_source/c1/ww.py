import math
import numpy as np
import h5py

# =============================================================================
# Process or Plot results
# =============================================================================

# Results
v=29.9792458
pi=math.acos(-1)

with h5py.File('output.h5', 'r') as f:
	x     = f['tally/grid/x'][:]
	x_mid = 0.5*(x[:-1]+x[1:])
	y     = f['tally/grid/y'][:]
	y_mid = 0.5*(y[:-1]+y[1:])
	X,Y = np.meshgrid(x_mid,y_mid)
	t     = f['tally/grid/t'][:]
	dt    = t[1:] - t[:-1]
	K     = len(dt)

	#cf = 1.0
	cf = 1 + pi*4E-2/v

	phi    = f['tally/flux/mean'][:]*100*cf
	phi_sd = f['tally/flux/sdev'][:]*100*cf
	T = f['runtime'][:]

for k in range(K):
	phi[k] /= dt[k] 			#Scalar Flux
	phi_sd[k] /= dt[k] 		#MC standard deviation

ww=phi
#ww /= phi[0].max().max() # used a fixed normalization constant
for wws in ww:
	wws /= wws.max().max() #normalize every time step
#	wws /= wws
np.savez("ww.npz",phi=ww)

for k in range(K):
	ww[k] = ww[k].transpose()

#write file for standard deviation
with open('ww.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, ww[k], fmt='%-10.4e')
