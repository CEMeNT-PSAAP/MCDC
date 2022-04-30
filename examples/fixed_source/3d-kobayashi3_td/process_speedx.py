import numpy as np
import h5py
import matplotlib.pyplot as plt

Np = np.array([100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000])
py = np.zeros_like(Np, dtype=float)
nb = np.zeros_like(Np, dtype=float)

for i in range(len(Np)):
    with h5py.File('speedx_n%i.h5'%(i+1), 'r') as f:
        t = f['runtime'][:]
        nb[i] = t
    with h5py.File('speedx_p%i.h5'%(i+1), 'r') as f:
        t = f['runtime'][:]
        py[i] = t

print(py/nb)
print(py/(nb-nb[0]))
plt.figure(figsize=(4,3))
plt.plot(Np, py, 'rs', fillstyle='none', label='Python')
plt.plot(Np, nb, 'bo', fillstyle='none', label='Numba')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# of histories')
plt.ylabel('Runtime [s]')
plt.grid()
plt.legend()
plt.savefig('kobay_speedupx.svg',dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()
