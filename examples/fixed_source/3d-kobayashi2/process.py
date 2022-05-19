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
ref = np.array([8.61696, 2.16123, 8.93437E-1, 4.77452E-1, 2.88719E-1, 1.88959E-1, 1.31026E-1, 9.49890E-2, 7.12403E-2, 5.44807E-2])
ref_sd = np.array([0.063,0.015,0.011,0.012,0.013,0.014,0.016,0.017,0.019,0.019])/100.0*ref
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('2A')
plt.show()

phi    = _phi[:,-1,0]
phi_sd = _phi_sd[:,-1,0]
ref = np.array([5.44807E-2, 6.58233E-3, 1.28002E-3, 4.13414E-4, 1.55548E-4, 6.02771E-5])
ref_sd = np.array([0.019,0.244,0.336,0.363,0.454,0.599])/100.0*ref
grid_mid = grid_mid[:6]
plt.plot(grid_mid,phi,'b-o', fillstyle='none', label='MC/DC')
plt.fill_between(grid_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.plot(grid_mid,ref,'r--x',label='Ref.')
plt.fill_between(grid_mid,ref-ref_sd,ref+ref_sd,alpha=0.2,color='r')
plt.legend()
plt.yscale('log')
plt.grid()
plt.title('2B')
plt.show()
