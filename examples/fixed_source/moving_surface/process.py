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

    phi    = f['tally/flux/mean'][:]
    phi_sd = f['tally/flux/sdev'][:]
    J      = f['tally/current/mean'][:,:,0]
    J_sd   = f['tally/current/sdev'][:,:,0]

# Flux - average
fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{k,j}$')
line1, = ax.plot([], [],'-b',label="MC")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, text
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()

# Flux - average
fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{J}_{k,j}$')
line1, = ax.plot([], [],'-b',label="MC")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,J[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,J[k,:]-J_sd[k,:],J[k,:]+J_sd[k,:],alpha=0.2,color='b')
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, text
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
