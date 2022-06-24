import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.linalg import expm
import matplotlib.animation as animation


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('SHEM-361.npz') as data:
    SigmaT     = data['SigmaT'] + data['SigmaC'] # /cm
    SigmaC     = data['SigmaC']
    SigmaS     = data['SigmaS']
    nuSigmaF_p = data['nuSigmaF_p']
    SigmaF     = data['SigmaF']
    nu_p       = data['nu_p']
    nu_d       = data['nu_d']
    chi_p      = data['chi_p']
    chi_d      = data['chi_d']
    G          = data['G']
    J          = data['J']
    E          = data['E']
    v          = data['v']
    lamd       = data['lamd']
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

# Time grid
N = 100
t = np.logspace(-10,2,N)

# Matrix and RHS source
A = np.zeros([G+J, G+J])
Q = np.zeros(G+J)

# Top-left [GxG]: phi --> phi
A[:G,:G] = SigmaS + nuSigmaF_p - np.diag(SigmaT)

# Top-right [GxJ]: C --> phi
A[:G,G:] = np.multiply(chi_d,lamd)

# Bottom-left [JxG]: phi --> C
A[G:,:G] = np.multiply(nu_d,SigmaF)

# bottom-right [JxJ]: C --> C
A[G:,G:] = -np.diag(lamd)

# Multiply with neutron speed
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(v), A[:G,:])

# Initial condition
PHI_init = np.zeros(G+J)
#PHI_init[:G] = v/G
PHI_init[G-2] = v[-2]

# Analytical particular solution
PHI_p = np.dot(np.linalg.inv(-A),Q)

# Allocate solution
PHI   = np.zeros([N,G+J])
C     = np.zeros((N,J))
n_tot = np.zeros(N)

# Analytical solution
for n in range(N):
    # Flux
    PHI_h    = np.dot(expm(AV*t[n]),(PHI_init - PHI_p))
    PHI[n,:] = PHI_h + PHI_p
    C[n,:]   = PHI[n,G:]
    n_tot[n] = sum(np.divide(PHI[n,:G],v))

# =============================================================================
# Plot results
# =============================================================================

plt.loglog(t,n_tot)
plt.show()

# Results
with h5py.File('output.h5', 'r') as f:
    phi = f['tally/flux-t/mean'][:]
    phi_sd = f['tally/flux-t/sdev'][:]
   
for n in range(N):
    PHI[n,:G]     = PHI[n,:G]*E_mid/dE
    phi[:,n+1]    *= E_mid/dE
    phi_sd[:,n+1] *= E_mid/dE

# Flux - t
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(E_mid[0], E_mid[-1]), ylim=((PHI[0,:G]).min(), (PHI[0,:G]).max()))
ax.grid()
ax.set_xlabel(r'$E$, MeV')
ax.set_ylabel(r'$E\phi(E)$')
ax.set_title(r'$\phi(E,t)$')
ax.set_xscale('log')
line1, = ax.step([], [],'-b',where='mid',label="MC")
line2, = ax.step([], [],'--r',where='mid',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(n):
    line1.set_data(E_mid,phi[:,n+1])
    ax.collections.clear()
    ax.fill_between(E_mid,phi[:,n+1]-phi_sd[:,n+1],phi[:,n+1]+phi_sd[:,n+1],alpha=0.2,color='b',step='mid')
    line2.set_data(E_mid,PHI[n,:G])
    ax.set_ylim([(PHI[n,:G]).min(), (PHI[n,:G]).max()])
    ax.legend(loc=6)
    text.set_text(r'$t = %.8f$ s'%(t[n]))
    return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=N)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
