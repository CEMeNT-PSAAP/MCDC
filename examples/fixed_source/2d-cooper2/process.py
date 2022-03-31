import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    x     = f['tally/grid/x'][:]
    x_mid = 0.5*(x[:-1]+x[1:])
    y     = f['tally/grid/y'][:]
    y_mid = 0.5*(y[:-1]+y[1:])
    X,Y = np.meshgrid(x_mid,y_mid)
    
    phi    = f['tally/flux/mean'][:]/(0.5**2)
    phi_sd = f['tally/flux/sdev'][:]/(0.5**2)

phi = np.log10(phi.transpose())
fig = plt.figure(figsize=(4,3))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,phi,color='w',edgecolors='k', lw=0.1)
ax.set_facecolor('w')
ax.azim = 50
ax.elev = 24
ax.dist = 9
ax.set_xlabel(r'$x$ [cm]')
ax.set_ylabel(r'$y$ [cm]')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'Log10 of scalar flux', rotation=90)
plt.show()
