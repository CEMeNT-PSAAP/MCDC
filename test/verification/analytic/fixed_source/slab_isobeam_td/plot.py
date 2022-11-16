import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import sys

from reference import reference

# Loda grid
output = sys.argv[1]
with h5py.File(output, 'r') as f:
    x = f['tally/grid/x'][:]
    t = f['tally/grid/t'][:]
    dx    = (x[1]-x[0])
    x_mid = 0.5*(x[:-1]+x[1:])
    dt    = t[1:] - t[:-1]
    K     = len(dt)
    J     = len(x_mid)
# Reference solution
phi_ref = reference(x,t)

# =============================================================================
# Animate results
# =============================================================================

with h5py.File(output, 'r') as f:
    phi    = f['tally/flux-t/mean'][1:]
    phi_sd = f['tally/flux-t/sdev'][1:]
for k in range(K):
    phi[k]    *= (0.5/dx)
    phi_sd[k] *= (0.5/dx)

# Animate
fig = plt.figure(figsize=(6,4))
ax = plt.axes(ylim=(5E-5, 4E-1))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{j}(t)$')
ax.set_yscale('log')
line1, = ax.plot([], [],'-b',label="MC/DC")
line2, = ax.plot([], [],'--r',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x_mid,phi_ref[k,:])
    text.set_text(r'$t = %.1f$ s'%(t[k+1]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()

