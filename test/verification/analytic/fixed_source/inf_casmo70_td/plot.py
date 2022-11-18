import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import matplotlib.animation as animation

from reference import reference

output = sys.argv[1]

# Load results
output = sys.argv[1]
with np.load('CASMO-70.npz') as data:
    E     = data['E']
    G     = data['G']
    speed = data['v']
    E_mid = 0.5*(E[1:]+E[:-1])
    dE    = E[1:]-E[:-1]
with h5py.File(output, 'r') as f:
    phi     = f['tally/flux-t/mean'][:]
    phi_sd  = f['tally/flux-t/sdev'][:]
    t       = f['tally/grid/t'][:]
    K       = len(t)-1
phi    = phi[:,1:]
phi_sd = phi_sd[:,1:]

# Neutron density
n    = np.zeros(K)
n_sd = np.zeros(K)
for k in range(K):
    n[k]    = np.sum(phi[:,k]/speed)
    n_sd[k] = np.linalg.norm(phi_sd[:,k]/speed)

# Reference solution
phi_ref, n_ref = reference(t)

# Neutron density
plt.plot(t[1:],n,'-b',label="MC")
plt.fill_between(t[1:],n-n_sd,n+n_sd,alpha=0.2,color='b')
plt.plot(t,n_ref,'--r',label="Ref.")
plt.xlabel(r'$t$, s')
plt.ylabel('Density')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title(r'$n_g(t)$')
plt.show()

phi_ref = phi_ref[1:]
for k in range(K):
    phi_ref[k,:] *= E_mid/dE
    phi[:,k]     *= E_mid/dE
    phi_sd[:,k]  *= E_mid/dE

# Flux - t
fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.grid()
ax.set_xlabel(r'$E$, MeV')
ax.set_ylabel(r'$E\phi(E)$')
ax.set_title(r'$\phi(E,t)$')
ax.set_xscale('log')
line1, = ax.step([], [],'-b',where='mid',label="MC")
line2, = ax.step([], [],'--r',where='mid',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):
    line1.set_data(E_mid,phi[:,k])
    ax.collections.clear()
    ax.fill_between(E_mid,phi[:,k]-phi_sd[:,k],phi[:,k]+phi_sd[:,k],alpha=0.2,color='b',step='mid')
    line2.set_data(E_mid,phi_ref[k,:G])
    ax.set_ylim([(phi_ref[k,:G]).min(), (phi_ref[k,:G]).max()])
    ax.legend()
    text.set_text(r'$t = %.8f$ s'%(t[k+1]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
