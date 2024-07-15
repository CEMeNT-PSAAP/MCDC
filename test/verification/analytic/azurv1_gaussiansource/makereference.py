import numpy as np
import h5py
import scipy
x = np.linspace(-20.5,20.5,202)
t = np.linspace(.5,19.5,20)
phi = np.empty((10001,201))
tavg = np.linspace(0,20,10001)
with h5py.File("../benchmarks.hdf5",'r') as f:
    for i in range(10001):
        key = 'gaussian_source/t = '+str(tavg[i])+'x0=0.5'
        phi[i,:] = f[key][1,:]
phiavg = np.empty((20,201))
for i in range(20):
    for j in range(201):
        phiavg[i,j] = scipy.integrate.simpson(y = phi[(500*i):(500*i+500),j],x = tavg[(500*i):(500+500*i)])
np.savez("reference.npz", x=x, t=t, phi=phiavg)
