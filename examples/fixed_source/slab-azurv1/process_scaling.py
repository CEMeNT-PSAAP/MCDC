import numpy as np
import h5py
import matplotlib.pyplot as plt

Np = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
t  = np.zeros_like(Np, dtype=float)

for i in range(len(Np)):
    with h5py.File('output_n%i.h5'%(i+1), 'r') as f:
        t[i] = f['runtime'][:]

plt.figure(figsize=(4,3))
plt.plot(Np, t, 'bo', fillstyle='none')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# of histories')
plt.ylabel('Runtime [s]')
plt.grid()
plt.legend()
plt.savefig('azurv_scale.svg',dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()
