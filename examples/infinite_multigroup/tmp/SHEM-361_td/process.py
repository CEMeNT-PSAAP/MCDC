import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.linalg import expm
import matplotlib.animation as animation



# =============================================================================
# Reference solution
# =============================================================================

# Load XS
data       = np.load('SHEM-361.npz')
SigmaT     = data['SigmaT']
nuSigmaF   = data['nuSigmaF']
SigmaS     = data['SigmaS']
SigmaF     = data['SigmaF']
v      = data['v']
nuSigmaF_p = data['nuSigmaF_p']
chi_d      = data['chi_d']
lamd       = data['lamd']
nu_d       = data['nu_d']
E          = data['E']
SigmaC     = data['SigmaC']

SigmaT += SigmaC*0.2595

G = len(SigmaT)
J = len(lamd)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

A = np.zeros([G+J,G+J])

# Top-left [GxG]: phi --> phi
A[:G,:G] = SigmaS + nuSigmaF_p - np.diag(SigmaT)

# Top-right [GxJ]: C --> phi
A[:G,G:] = np.multiply(chi_d,lamd)

# Bottom-left [JxG]: phi --> C
A[G:,:G] = np.multiply(nu_d,SigmaF)

# bottom-right [JxJ]: C --> C
A[G:,G:] = -np.diag(lamd)

# Multiply with neutron v
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(v), A[:G,:])

# Particular solution
Q = np.zeros(G+J)
PHI_p = np.dot(np.linalg.inv(-A),Q)

# Initial condition
PHI_init = np.zeros(G+J)
PHI_init[G-1] = v[-1]

# Time grid
N = 50
t = np.logspace(-9,-0.5,N)

# Solve
PHI = np.zeros([G+J,N])
n_tot = np.zeros(N)
for n in range(N):
    # Flux
    PHI_h    = np.dot(expm(AV*t[n]),(PHI_init - PHI_p))
    PHI[:,n] = PHI_h + PHI_p
    n_tot[n] = sum(np.divide(PHI[:G,n],v))

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi    = f['tally/flux-edge/mean'][:]
    phi_sd = f['tally/flux-edge/sdev'][:]
# Normalize eigenvector
n = np.zeros(N)
n_sd = np.zeros(N)
for i in range(N): 
    n[i] = sum(np.divide(phi[i],v))
    n_sd[i] = sum(np.divide(phi_sd[i],v)**2)**0.5

plt.plot(t,n,'b-',label="MC")
plt.fill_between(t,n-n_sd,n+n_sd,alpha=0.2,color='b')
plt.plot(t,n_tot,'r--',label='Ref.')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$n(t)$')
plt.grid()
plt.show()

for n in range(N):
    PHI[:G,n] = PHI[:G,n]/dE*E_mid
    phi[n,:]  = phi[n,:]/dE*E_mid
    phi_sd[n,:]  = phi_sd[n,:]/dE*E_mid
    plt.plot(E_mid,PHI[:G,n])
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
plt.clf()
plt.close()

fig = plt.figure(figsize=(6,4))
ax = plt.gca()
ax.grid()
ax.set_xlabel(r'$E$')
ax.set_ylabel(r'$E\phi(E)$')    
ax.set_xlim(left=xmin, right=xmax)
ax.set_ylim(bottom=ymin, top=ymax)
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
ax.set_xscale('log')
ax.set_yscale('log')
def animate(k):        
    line1.set_data(E_mid,phi[k,:])
    ax.collections.clear()
    ax.fill_between(E_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    line2.set_data(E_mid,PHI[:G,k])
    text.set_text(r'$t=%.1e$ s'%(t[k]))
    return line1, line2
    #return line1, line2, text        
simulation = animation.FuncAnimation(fig, animate, frames=N)
writervideo = animation.FFMpegWriter(fps=30)
plt.show()
