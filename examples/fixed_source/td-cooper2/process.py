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

#with np.load('ww.npz') as data:
#	ww = data['phi']

Nx = len(x_mid)
Ny = len(y_mid)

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
FOM = 1/np.sum(n)/rel_var			#piecewise figure of merit
FOM[n==0]=0

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
with open('phi.txt', 'w') as outfile:
    print_var(outfile, phi)

#write file for standard deviation
with open('phi_sd.txt', 'w') as outfile:
    print_var(outfile, phi_sd)

#write file for rel var
with open('relvar.txt', 'w') as outfile:
    print_var(outfile, rel_var)

#write file for local Figure of Merit
with open('FOM.txt', 'w') as outfile:
    print_var(outfile, FOM)

#write file for n
with open('n.txt', 'w') as outfile:
    print_var(outfile, n)

#write file for alpha
with open('alpha.txt', 'w') as outfile:
    print_var(outfile, alpha)
