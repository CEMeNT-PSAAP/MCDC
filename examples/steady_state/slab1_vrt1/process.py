import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load grids
with h5py.File('output_N=1.h5', 'r') as f:
    x = f['tally/spatial_grid'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# XS
SigmaT1 = 1.0
SigmaT2 = 1.5
SigmaT3 = 2.0

# Spatial average flux
phi_ref = -(1-np.exp(SigmaT2*dx))*np.exp(-SigmaT2*x[1:21])/SigmaT2/dx
phi_ref = np.append(phi_ref, np.exp(-SigmaT2*2.0)*-(1-np.exp(SigmaT3*dx))*np.exp(-SigmaT3*x[1:21])/SigmaT3/dx)
phi_ref = np.append(phi_ref, np.exp(-(SigmaT2+SigmaT3)*2.0)*-(1-np.exp(SigmaT1*dx))*np.exp(-SigmaT1*x[1:21])/SigmaT1/dx)

# Spatial grid flux
phi_face_ref = np.exp(-SigmaT2*x[1:21])
phi_face_ref = np.append(phi_face_ref, np.exp(-SigmaT2*2.0)*np.exp(-SigmaT3*x[1:21]))
phi_face_ref = np.append(phi_face_ref, np.exp(-SigmaT2*2.0)*np.exp(-SigmaT3*2.0)*np.exp(-SigmaT1*x[1:21]))

# =============================================================================
# Plot results
# =============================================================================

# Cases to process and error container
N_hist_list = np.logspace(0,7,15).astype(int)
err_phi = []
err_phi_face = []

for N_hist in N_hist_list:
    # Neutron flux
    with h5py.File('output_N=%i.h5'%N_hist, 'r') as f:
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][1:]
        phi_face_sd = f['tally/flux-face/sdev'][1:]
    err_phi.append(np.linalg.norm(phi_ref-phi))
    err_phi_face.append(np.linalg.norm(phi_face_ref-phi_face))

    # Plot
    if N_hist%10 == 0:
        # Solution
        plt.plot(x_mid,phi,'-b',label="MC")
        plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
        plt.plot(x_mid,phi_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Flux')
        plt.ylim([1E-4,1E0])
        plt.grid()
        plt.yscale('log')
        plt.legend()
        plt.title(r'$\bar{\phi}_i$, $N=%i$'%N_hist)
        plt.savefig('phi_N=%i.png'%N_hist,dpi=1200)
        plt.clf()

        plt.plot(x[1:],phi_face,'-b',label="MC")
        plt.fill_between(x[1:],phi_face-phi_face_sd,phi_face+phi_face_sd,alpha=0.2,color='b')
        plt.plot(x[1:],phi_face_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Flux')
        plt.ylim([1E-4,1E0])
        plt.grid()
        plt.yscale('log')
        plt.legend()
        plt.title(r'$\phi(x)$, $N=%i$'%N_hist)
        plt.savefig('phi_face_N=%i.png'%N_hist,dpi=1200)
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
        plt.title(r'$\bar{\phi}_i$ Convergence')
        plt.savefig('phi_convergence.png',dpi=1200)
        plt.clf()

        plt.plot(N,err_phi_face,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('Error 2-norm')
        plt.grid()
        plt.legend()
        plt.title(r'$\phi(x)$ Convergence')
        plt.savefig('phi_face_convergence.png',dpi=1200)
        plt.clf()
