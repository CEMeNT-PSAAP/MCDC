import matplotlib.pyplot as plt
import numpy as np
import h5py

grid = np.linspace(0.0,100.0,11)
grid_mid = 0.5*(grid[:-1] + grid[1:])
grid_size = grid[1] - grid[0]

with h5py.File('output.h5', 'r') as f:
    _phi    = f['tally/flux/mean'][:]
    _phi_sd = f['tally/flux/sdev'][:]

phi    = _phi[0,:,0]
phi_sd = _phi_sd[0,:,0]
ref = np.array([8.2926, 1.87028, 7.13986E-1, 3.84685E-1, 2.53984E-1, 1.3722E-1, 
                4.65913E-2, 1.58766E-2, 5.47036E-3, 1.85082E-3])
ref_sd = np.array([0.021,0.005,0.003,0.004,0.006,0.073,0.117,0.197,0.343,0.619])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('1A')
plt.show()

phi    = np.zeros_like(phi)
phi_sd  = np.zeros_like(phi_sd)
for i in range(len(grid_mid)):
    phi[i] = _phi[i,i,i]
    phi_sd[i] = _phi_sd[i,i,i]
ref = np.array([8.29260, 6.63233E-1, 2.68828E-1, 1.56683E-1, 1.04405E-1, 3.02145E-2, 4.06555E-3, 5.86124E-4, 8.66059E-5, 1.12982E-5])
ref_sd = np.array([0.021,0.004,0.003,0.005,0.011,0.061,0.074,0.116,0.198,0.383])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('1B')
plt.show()

phi    = _phi[:,5,0]
phi_sd = _phi_sd[:,5,0]
ref = np.array([1.37220E-1, 1.27890E-1, 1.13582E-1, 9.59578E-2, 7.82701E-2, 5.67030E-2, 1.88631E-2, 6.46624E-3, 2.28099E-3, 7.93924E-4])
ref_sd = np.array([0.073,0.076,0.080,0.088,0.094,0.111,0.189,0.314,0.529,0.890])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('1C')
plt.show()
