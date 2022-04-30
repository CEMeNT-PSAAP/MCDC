import numpy as np
import h5py
import matplotlib.pyplot as plt

Np = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
t1 = np.zeros_like(Np, dtype=float)
t2 = np.zeros_like(Np, dtype=float)
ef1 = np.zeros_like(t1)
ef2 = np.zeros_like(t2)

for i in range(len(Np)):
    with h5py.File('slab-azurv1/old/output_n%i.h5'%(i+1), 'r') as f:
        t1[i] = f['runtime'][:]
    with h5py.File('3d-kobayashi3_td/old/output_n%i.h5'%(i+1), 'r') as f:
        t2[i] = f['runtime'][:]

ef1 = t1[0]/t1
ef2 = t2[0]/t2

plt.figure(figsize=(4,3))
plt.plot(Np, ef1, 'bo', fillstyle='none', label='AZURV1')
plt.plot(Np, ef2, 'rs', fillstyle='none', label='TD-Kobayashi')
plt.xscale('log')
plt.xlabel('# of MPI processes')
plt.ylabel('Weak scaling efficiency')
plt.grid()
plt.legend()
plt.savefig('tst2_scale.svg',dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()
