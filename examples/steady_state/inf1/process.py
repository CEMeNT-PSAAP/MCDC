import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('XS.npz') as data:
    speeds = data['v']        # cm/s
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu     = data['nu']
    E      = data['E']        # eV
G     = len(speeds)
E_mid = 0.5*(E[:-1] + E[1:])
dE    = E[1:] - E[:-1]

# Augment with uniform leakage XS
SigmaL  = 0.24 # /cm
SigmaT += SigmaL

# Analytical solution
nuSigmaF = SigmaF.dot(np.diag(nu))
A = np.diag(SigmaT) - SigmaS - nuSigmaF
b = np.zeros(G)
b[-1] = 1.0
phi_ref = np.linalg.solve(A,b)
phi_ref = np.divide(phi_ref,dE)
phi_ref = np.multiply(phi_ref,E_mid)

# =============================================================================
# Plot results
# =============================================================================

# Cases to process and error container
N_hist_list = np.logspace(0,7,15).astype(int)
err_phi = []

for N_hist in N_hist_list:
    # Neutron flux
    with h5py.File('output_N=%i.h5'%N_hist, 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              
    err_phi.append(np.linalg.norm(phi_ref-phi))
    
    # Plot
    if N_hist%10 == 0:
        # Solution
        plt.plot(E_mid,phi,'-b',label="MC")
        plt.fill_between(E_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
        plt.plot(E_mid,phi_ref,'--r',label='ref.')
        plt.xscale('log')
        plt.xlabel(r'$E$, eV')
        plt.ylabel(r'$E\phi(E)$')
        plt.grid()
        plt.legend()
        plt.title(r'$N=%i$'%N_hist)
        plt.savefig('N=%i.png'%N_hist,dpi=1200)
        plt.clf()
    
        # Convergence
        N   = N_hist_list[:len(err_phi)]
        fac = err_phi[-1]/(1/np.sqrt(N[-1]))
        plt.plot(N,err_phi,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('Error 2-norm')
        plt.grid()
        plt.legend()
        plt.savefig('convergence.png',dpi=1200)
        plt.clf()
