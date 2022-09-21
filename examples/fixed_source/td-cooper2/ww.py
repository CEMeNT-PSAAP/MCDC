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
	t     = f['tally/grid/t'][:]
	dt    = t[1:] - t[:-1]
	K     = len(dt)

	cf = 5*5*2+pi*4E-10*400/v

	phi    = f['tally/flux/mean'][:]*4*cf
	phi_sd = f['tally/flux/sdev'][:]*4*cf
	T = f['runtime'][:]

#distance from corner of source
d1 = np.sqrt((X-0.25-5)*(X-0.25-5)+(Y-0.25-5)*(Y-0.25-5))
#distance from top edge
d2 = np.abs(Y-0.25-5)
#distance from right edge of source
d3 = np.abs(X-0.25-5)

#composite distance
d = np.zeros_like(X)
d[10:,10:]=d1[10:,10:]
d[10:,:10]=d2[10:,:10]
d[:10,10:]=d3[:10,10:]

waved=t[1:]*v

thresh=np.zeros_like(phi)
for k in range(K):
	thresh[k] = d<waved[k]

for k in range(K):
	phi[k] /= dt[k] 			#Scalar Flux
	phi_sd[k] /= dt[k] 		#MC standard deviation

ww=phi
#ww /= phi[0].max().max() # used a fixed normalization constant
for k in range(K):
	wws = ww[k]
	wws[wws==0] = np.min(wws[wws>0]) #ensure no zeros
	wws /= wws.max().max() #normalize every time step
	wws=wws*thresh[k]+(1-thresh[k])*wws[30,30]
#	wws /= wws
	ww[k] = wws

np.savez("ww.npz",phi=ww)

for k in range(K):
	ww[k] = ww[k].transpose()

#write file for standard deviation
with open('ww.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, ww[k], fmt='%-10.4e')
