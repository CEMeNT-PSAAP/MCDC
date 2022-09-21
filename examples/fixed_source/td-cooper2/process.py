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

with np.load('ww.npz') as data:
	ww = data['phi']

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
rel_var_t = phit_sd*phit_sd/phit/phit #MC relative variance
FOM_t = 1/T/rel_var_t		#piecewise figure of merit

#write file for phi
with open('phi.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, phi[k], fmt='%-10.4e')

#write file for standard deviation
with open('phi_sd.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, phi_sd[k], fmt='%-10.4e')

#write file for local Figure of Merit
with open('FOM.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, FOM[k], fmt='%-10.4e')

#write file for n
with open('n.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, n[k], fmt='%-10.4e')

#write file for alpha
with open('alpha.txt', 'w') as outfile:
    for k in range(K):
        outfile.write("Time Step "+str(k+1)+"\n")
        np.savetxt(outfile, alpha[k], fmt='%-10.4e')

#write file for phi_t
with open('phi-t.txt', 'w') as outfile:
    for k in range(K+1):
        outfile.write("Time Step "+str(k)+"\n")
        np.savetxt(outfile, phit[k], fmt='%-10.4e')

#write file for phi_t standard deviation
with open('phi-t_sd.txt', 'w') as outfile:
    for k in range(K+1):
        outfile.write("Time Step "+str(k)+"\n")
        np.savetxt(outfile, phit_sd[k], fmt='%-10.4e')

#write file for local Figure of Merit
with open('FOM-t.txt', 'w') as outfile:
    for k in range(K+1):
        outfile.write("Time Step "+str(k)+"\n")
        np.savetxt(outfile, FOM_t[k], fmt='%-10.4e')

#write file for n
with open('n-t.txt', 'w') as outfile:
    for k in range(K+1):
        outfile.write("Time Step "+str(k)+"\n")
        np.savetxt(outfile, nt[k], fmt='%-10.4e')

