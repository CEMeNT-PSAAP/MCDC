import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# =============================================================================
# Plot
# =============================================================================

# Load output
with h5py.File('output.h5', 'r') as f:
    x      = f['tally/grid/x'][:]
    x_mid  = 0.5*(x[:-1]+x[1:])
    dx     = x[1:]-x[:-1]
    t      = f['tally/grid/t'][:]
    dt     = t[1:]-t[:-1]
    K      = len(t)-1

    phi    = f['tally/flux-t/mean'][:]
    phi_sd = f['tally/flux-t/sdev'][:]

# Flux - average
fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{k,j}$')
line1, = ax.plot([], [],'-b',label="MC")
asp1 = ax.axvspan(0, 2, facecolor='r', alpha=0.1)
asp2 = ax.axvspan(2, 6, facecolor='g', alpha=0.1)
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()       

def s(t):
    if t < 18.0:
        return 2.0
    elif t < 24.0:
        return 2.0 + 0.5*(t-18.0)
    else:
        return 1.0
def animate(k):        
    global asp1, asp2
    k = k+1
    ax.collections.clear()
    asp1.remove()
    asp2.remove()
    line1.set_data(x_mid,phi[k,:])
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    asp1 = ax.axvspan(0, s(t[k]), facecolor='r', alpha=0.2)
    asp2 = ax.axvspan(s(t[k]), 6.0, facecolor='g', alpha=0.2)
    text.set_text(r'$t=%.1f$ s'%(t[k]))
    return line1, text
simulation = animation.FuncAnimation(fig, animate, frames=K)
#simulation.save('animation.gif', writer='imagemagick', fps=60)
plt.show()
