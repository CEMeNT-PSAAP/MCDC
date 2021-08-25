import numpy as np
import matplotlib.pyplot as plt
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
# Plot results
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

# Plot
for k in range(K):
    plt.plot(x_mid,phi_edge[k],'-ob',label="MC")
    plt.fill_between(x_mid,phi_edge[k]-phi_edge_sd[k],phi_edge[k]+phi_edge_sd[k],alpha=0.2,color='b')
    plt.plot(x_mid,phi_edge_ref[k],'--xr',label="ref.")
    plt.xlabel(r'$x$, cm')
    plt.ylabel('Flux')
    plt.xlim([max(x[0],-t[k+1]*1.6),min(t[k+1]*1.6,x[-1])])
    plt.grid()
    plt.legend()
    plt.title(r'$\bar{\phi}_i(t=%.1f)$'%t[k+1])
    plt.show()
                
    plt.plot(x,phi_face[k],'-ob',label="MC")
    plt.fill_between(x,phi_face[k]-phi_face_sd[k],phi_face[k]+phi_face_sd[k],alpha=0.2,color='b')
    plt.plot(x,phi_face_ref[k],'--xr',label="ref.")
    plt.xlabel(r'$x$, cm')
    plt.ylabel('Flux')
    plt.xlim([max(x[0],-t[k+1]*1.6),min(t[k+1]*1.6,x[-1])])
    plt.grid()
    plt.legend()
    plt.title(r'$\bar{\phi}_k(x)$, $t_k\in[%.1f,%.1f]$'%(t[k],t[k+1]))
    plr.show()
    
    plt.plot(x_mid,phi[k],'-ob',label="MC")
    plt.fill_between(x_mid,phi[k]-phi_sd[k],phi[k]+phi_sd[k],alpha=0.2,color='b')
    plt.plot(x_mid,phi_ref[k],'--xr',label="ref.")
    plt.xlabel(r'$x$, cm')
    plt.ylabel('Flux')
    plt.xlim([max(x[0],-t[k+1]*1.6),min(t[k+1]*1.6,x[-1])])
    plt.grid()
    plt.legend()
    plt.title(r'$\bar{\phi}_{i,k}$, $t_k\in[%.1f,%.1f]$'%(t[k],t[k+1]))
    plt.show()
