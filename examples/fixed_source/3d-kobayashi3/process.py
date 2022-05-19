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
ref = np.array([8.61578, 2.16130, 8.93784E-1, 4.78052E-1, 2.89424E-1, 1.92698E-1, 1.04982E-1, 3.37544E-2, 1.08158E-2, 3.39632E-3])
ref_sd = np.array([0.044,0.010,0.008,0.008,0.009,0.010,0.077,0.107,0.163,0.275])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('3A')
plt.show()

phi    = _phi[:,5,0]
phi_sd = _phi_sd[:,-1,0]
ref = np.array([1.92698E-1, 6.72147E-2, 2.21799E-2, 9.90646E-3, 3.39066E-3, 1.05629E-3])
ref_sd = np.array([0.010,0.019,0.028,0.033,0.195,0.327])/100.0*ref
grid_mid = grid_mid[:6]
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('3B')
plt.show()

phi    = _phi[:,-1,3]
phi_sd = _phi_sd[:,-1,0]
ref = np.array([3.44804E-4, 2.91825E-4, 2.05793E-4, 2.62086E-4, 1.05367E-4, 4.44962E-5])
ref_sd = np.array([0.793,0.659,0.529,0.075,0.402,0.440])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('3C')
plt.show()
