import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from matplotlib import colors

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    fis_avg = f['tally/fission/mean'][:]
    fis_sd  = f['tally/fission/sdev'][:]
    t       = f['tally/grid/t'][:]
    x       = f['tally/grid/x'][:]
    z       = f['tally/grid/z'][:]
dx = x[1] - x[0]
dz = z[1] - z[0]
dV = dx*dx*dz

t_mid = 0.5*(t[:-1] + t[1:])
dt    = (t[1:] - t[:-1])

# Normalize
norm = np.sum(fis_avg[:,0])/dt[0]
fis_avg /= norm
fis_sd  /= norm
fis_    = np.zeros_like(fis_avg[0])
fis__sd = np.zeros_like(fis_avg[0])
for i in range(7):
    fis_    += fis_avg[i]
    fis__sd += np.square(fis_sd[i])
fis__sd = np.sqrt(fis__sd)
fis_ /= dt
fis__sd /= dt

# Reference
with h5py.File('reference.h5', 'r') as f:
    fis_avg_ref = f['tally/fission/mean'][:]
    fis_sd_ref  = f['tally/fission/sdev'][:]

# Normalize
norm = np.sum(fis_avg_ref[:,0])/dt[0]
fis_avg_ref /= norm
fis_sd_ref  /= norm
fis__ref    = np.zeros_like(fis_avg_ref[0])
fis__sd_ref = np.zeros_like(fis_avg_ref[0])
for i in range(7):
    fis__ref    += fis_avg_ref[i]
    fis__sd_ref += np.square(fis_sd_ref[i])
fis__sd_ref = np.sqrt(fis__sd_ref)
fis__ref /= dt
fis__sd_ref /= dt


plt.plot(t_mid,fis__ref,'-m',fillstyle='none',label='Ref. (MC)')
plt.fill_between(t_mid,fis__ref-fis__sd_ref,fis__ref+fis__sd_ref,alpha=0.2,
                 color='m')
plt.plot(t_mid,fis_,'ok',fillstyle='none',label='MC')
plt.fill_between(t_mid,fis_-fis__sd,fis_+fis__sd,alpha=0.2,color='k')
plt.yscale('log')
plt.ylabel('Normalized fission rate')
plt.xlabel('time [s]')
plt.axvspan(0, 5, facecolor='gray', alpha=0.2)
plt.axvspan(5, 10, facecolor='green', alpha=0.2)
plt.axvspan(10, 15, facecolor='red', alpha=0.2)
plt.axvspan(15, 20, facecolor='blue', alpha=0.2)
plt.annotate('Phase 1', (2.5, .3), color='black', ha='center', va='center',
             backgroundcolor='white')
plt.annotate('Phase 2', (7.5, .3), color='black', ha='center', va='center',
             backgroundcolor='white')
plt.annotate('Phase 3', (12.5,.3), color='black', ha='center', va='center',
             backgroundcolor='white')
plt.annotate('Phase 4', (17.5, 4.0), color='black', ha='center', va='center',
             backgroundcolor='white')
plt.xlim([0.0,20.0])
plt.ylim([0.09, 200.0])
plt.grid(which='both')
plt.legend()
plt.show()
