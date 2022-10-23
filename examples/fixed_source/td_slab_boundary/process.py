import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation
import h5py

from scipy.integrate import quad
from scipy.special   import exp1


# =============================================================================
# Reference solution
# =============================================================================

# Parameters
SigmaT = 0.25
v      = 1.0
T      = 20.0

# Point-wise solution
def phi_(x,t):
    if x > v*t:
        return 0.0
    else:
        return 1.0/T * (SigmaT*x*(exp1(SigmaT*v*t) - exp1(SigmaT*x)) + \
                        np.e**(-SigmaT*x) - x/(v*t)*np.e**(-SigmaT*v*t))

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/grid/x'][:]
    t = f['tally/grid/t'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
dt    = t[1:] - t[:-1]
K     = len(dt)
J     = len(x_mid)

# Cell-average, time-edge solution
phi_ref = np.zeros([K,J])
for k in range(K):
    for j in range(J):
        x1 = x[j]
        x2 = x[j+1]
        phi_ref[k,j] = quad(phi_,x1,x2,args=(t[k+1]))[0]/(x2-x1)

# =============================================================================
# Animate results
# =============================================================================

with h5py.File('output.h5', 'r') as f:
    phi    = f['tally/flux-t/mean'][1:]
    phi_sd = f['tally/flux-t/sdev'][1:]
for k in range(K):
    phi[k]    *= (0.5/dx)
    phi_sd[k] *= (0.5/dx)

# Animate
fig = plt.figure(figsize=(6,4))
ax = plt.axes(ylim=(5E-5, 2E-1))
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
simulation = animation.FuncAnimation(fig, animate, interval=10, frames=K)
#simulation.save('isotropic_boundary.gif', savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0})
plt.show()

