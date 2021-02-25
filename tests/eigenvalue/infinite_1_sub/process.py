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
A = np.dot(np.linalg.inv(np.diag(SigmaT) - SigmaS),nuSigmaF)
w,v = np.linalg.eig(A)
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]
k_exact = w[0]
phi_exact = v[:,0]
if phi_exact[0] < 0.0:
    phi_exact *= -1
phi_exact[:] /= np.sum(phi_exact[:])
phi_exact = phi_exact/dE*E_mid

# =============================================================================
# Plot results
# =============================================================================

# Cases to process and error container
N_hist_list = np.logspace(2,4,5).astype(int)
err_phi = []
err_k = []

for N_hist in N_hist_list:
    # Neutron flux
    with h5py.File('output_N=%i.h5'%N_hist, 'r') as f:
        phi = f['tally/flux/mean'][:]
        k   = f['keff'][:]

    # Normalize eigenvector
    for i in range(len(phi)): 
        phi[i][:] /= np.sum(phi[i][:])
        phi[i][:]/dE*E_mid

    # Get average
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
    phi_avg = phi_avg/dE*E_mid
    phi_sd  = phi_sd/dE*E_mid
    
    err_phi.append(np.linalg.norm(phi_exact-phi_avg))
    err_k.append(np.linalg.norm(k_exact-k_avg))

    # Plot
    if N_hist%10 == 0:
        N_iter = len(k)
        plt.plot(np.arange(2,N_iter+1),k[1:],'-b',label='MC')
        plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_exact,'--r',label='analytical')
        plt.plot(np.arange(2,N_iter+1),np.ones(N_iter-1)*k_avg,':g',label='MC-avg')
        plt.fill_between(np.arange(2,N_iter+1),np.ones(N_iter-1)*(k_avg-k_sd),np.ones(N_iter-1)*(k_avg+k_sd),alpha=0.2,color='g')
        plt.xlabel('Iteration #')
        plt.ylabel(r'$k$')
        plt.grid()
        plt.legend()
        plt.title(r'$N=%i$'%N_hist)
        plt.savefig('k_N=%i.png'%N_hist,dpi=1200)
        plt.clf()

        plt.plot(E_mid,phi_avg,'-b',label="MC")
        plt.fill_between(E_mid,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b')
        plt.plot(E_mid,phi_exact,'--r',label='analytical')
        plt.xscale('log')
        plt.xlabel(r'$E$, eV')
        plt.ylabel(r'$E\phi(E)$')
        plt.grid()
        plt.legend()
        plt.title(r'$N=%i$'%N_hist)
        plt.savefig('phi_N=%i.png'%N_hist,dpi=1200)
        plt.clf()
    
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
        plt.savefig('phi_convergence.png',dpi=1200)
        plt.clf()

        N   = N_hist_list[:len(err_k)]
        fac = err_k[-1]/(1/np.sqrt(N[-1]))
        plt.plot(N,err_k,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('errkor 2-norm')
        plt.grid()
        plt.legend()
        plt.savefig('k_convergence.png',dpi=1200)
        plt.clf()