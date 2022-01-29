import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad
import matplotlib.animation as animation


# =============================================================================
# Plot results
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/spatial_grid'][:]
    t = f['tally/time_grid'][:]

dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
dt    = t[1:] - t[:-1]
K     = len(dt)
J     = len(x_mid)

# Results
with h5py.File('output.h5', 'r') as f:
    phi         = f['tally/flux-edge/mean'][:]/dx
    phi_sd      = f['tally/flux-edge/sdev'][:]/dx


fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-0.1, 11.1), ylim=(np.min(phi), np.max(phi)))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux $\bar{\phi}_i(t)$')    
line1, = ax.plot([], [],'-b',label="MC")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    text.set_text(r'$t=%.1f$ s'%(t[k+1]))
    return line1, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=4)
plt.show()
