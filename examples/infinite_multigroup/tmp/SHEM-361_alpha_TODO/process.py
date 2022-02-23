import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
data       = np.load('SHEM-361.npz')
SigmaT     = data['SigmaT']
nuSigmaF   = data['nuSigmaF']
SigmaS     = data['SigmaS']
SigmaF     = data['SigmaF']
speed      = data['v']
nuSigmaF_p = data['nuSigmaF_p']
chi_d      = data['chi_d']
lamd       = data['lamd']
nu_d       = data['nu_d']
E          = data['E']

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

# Multiply with neutron speed
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(speed), A[:G,:])
w,v = np.linalg.eig(AV)
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]
alpha_exact = w[0]
phi_exact = v[:,0][:G]
phi_exact /= np.sum(phi_exact)
phi_exact = phi_exact/dE*E_mid

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File('output.h5', 'r') as f:
    phi = f['tally/flux/mean'][:]
    k   = f['keff'][:]
    
# Normalize eigenvector
for i in range(len(phi)): 
    phi[i][:] /= np.sum(phi[i][:])
    phi[i][:]  = phi[i][:]/dE*E_mid

# Get average (10 passive cycles)
phi_avg = np.zeros_like(phi[0])
phi_sd  = np.zeros_like(phi[0])
k_avg = 0.0
k_sd  = 0.0
for i in range(10,len(phi)): 
    phi_avg += phi[i][:]
    phi_sd  += np.square(phi[i][:])
    k_avg += k[i]
    k_sd  += k[i]**2
phi_avg /= 100    
phi_sd  = np.sqrt((phi_sd/100 - np.square(phi_avg))/(100-1))
k_avg /= 100    
k_sd  = np.sqrt((k_sd/100 - np.square(k_avg))/(100-1))
phi_avg = phi_avg
phi_sd  = phi_sd
    
# Plot
N_iter = len(k)
plt.plot(np.arange(2,N_iter+1),k[1:],'-b',label='MC')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*alpha_exact,'--r',label='analytical')
plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_avg,':g',label='MC-avg')
plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(k_avg-k_sd),np.ones(N_iter-1)*(k_avg+k_sd),alpha=0.2,color='g')
plt.xlabel('Iteration #')
plt.ylabel(r'$k$')
plt.grid()
plt.legend()
plt.show()

plt.plot(E_mid,phi_avg,'-b',label="MC")
plt.fill_between(E_mid,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
plt.plot(E_mid,phi_exact,'--r',label='analytical')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()
