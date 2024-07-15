import numpy as np
import h5py
import scipy
x = np.linspace(-20.5,20.5,202)
t = np.linspace(.5,19.5,20)
phi = np.empty((100001,201))
tavg = np.linspace(0,20,100001)
with h5py.File("../benchmarks.hdf5",'r') as f:
    for i in range(100001):
        key = 'square_IC/t = '+str(tavg[i])+'x0=0.5'
        phi[i,:] = f[key][1,:]
phiavg = np.empty((20,201))
for i in range(20):
    for j in range(201):
        if i == 0:
            if np.fabs(x[j])-.25<0:
                phiavg[i,j] = scipy.integrate.simpson(y = phi[(5000*i):(5000*i+5000),j],x = tavg[(5000*i):(5000+5000*i)])
            else:
                phiavg[i,j] = scipy.integrate.simpson(y = phi[(5000*i):(5000*i+5000),j],x = tavg[(5000*i):(5000+5000*i)])
        else:
            phiavg[i,j] = scipy.integrate.simpson(y = phi[(5000*i):(5000*i+5000),j],x = tavg[(5000*i):(5000+5000*i)])
np.savez("reference.npz", x=x, t=t, phi=phiavg)
