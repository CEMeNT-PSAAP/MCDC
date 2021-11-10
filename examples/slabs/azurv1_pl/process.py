import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py


# =============================================================================
# Reference solution (SS)
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

data = np.load('azurv1_pl.npz')
phi_edge_ref = data['phi_edge']
phi_face_ref = data['phi_face']
phi_ref = data['phi']

# =============================================================================
# Animate results
# =============================================================================

with h5py.File('output.h5', 'r') as f:
    phi         = f['tally/flux/mean'][:]/dx
    phi_sd      = f['tally/flux/sdev'][:]/dx
    phi_edge    = f['tally/flux-edge/mean'][:]/dx
    phi_edge_sd = f['tally/flux-edge/sdev'][:]/dx
    phi_face    = f['tally/flux-face/mean'][:]
    phi_face_sd = f['tally/flux-face/sdev'][:]
for k in range(K):
    phi[k] /= dt[k]
    phi_sd[k] /= dt[k]
    phi_face[k] /= dt[k]
    phi_face_sd[k] /= dt[k]

# Flux-edge
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux $\bar{\phi}_i(t)$')    
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,phi_edge[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi_edge[k,:]-phi_edge_sd[k,:],phi_edge[k,:]+phi_edge_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x_mid,phi_edge_ref[k,:])
    text.set_text(r'$t=%.1f$ s'%(t[k+1]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()

# Flux-face
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux $\bar{\phi}_k(x)$')    
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x,phi_face[k,:])
    ax.collections.clear()
    ax.fill_between(x,phi_face[k,:]-phi_face_sd[k,:],phi_face[k,:]+phi_face_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x,phi_face_ref[k,:])
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()

# Flux
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux $\bar{\phi}_{k,j}$')
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x_mid,phi_ref[k,:])
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
