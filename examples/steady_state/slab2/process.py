import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad


# =============================================================================
# Reference solution
# =============================================================================

# Load grids
with h5py.File('output_N=1.h5', 'r') as f:
    x = f['tally/spatial_grid'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# Parameters
SigmaT3 = 1.0
SigmaT1 = 1.5
SigmaT2 = 2.0
q1 = 1.0/6.0/2
q2 = 1.0/6.0/2
q3 = 1.0/6.0/2
x1 = 2.0
x2 = 4.0
x3 = 6.0
tau1 = SigmaT1*x1
tau2 = SigmaT2*(x2-x1)
tau3 = SigmaT3*(x3-x2)

# Angular flux
def psi1_plus(mu,x):
    return q1/SigmaT1*(1.0 - np.exp(-SigmaT1*x/mu))
def psi2_plus(mu,x):
    return (psi1_plus(mu,x1) - q2/SigmaT2)*np.exp(-SigmaT2*(x-x1)/mu) + q2/SigmaT2
def psi3_plus(mu,x):
    return (psi2_plus(mu,x2) - q3/SigmaT3)*np.exp(-SigmaT3*(x-x2)/mu) + q3/SigmaT3
def psi1_minus(mu,x):
    return (psi2_minus(mu,x1) - q1/SigmaT1)*np.exp(-SigmaT1*(x1-x)/np.abs(mu)) + q1/SigmaT1
def psi2_minus(mu,x):
    return (psi3_minus(mu,x2) - q2/SigmaT2)*np.exp(-SigmaT2*(x2-x)/np.abs(mu)) + q2/SigmaT2
def psi3_minus(mu,x):
    return q3/SigmaT3*(1.0 - np.exp(-SigmaT3*(x3-x)/np.abs(mu)))

# Flux
def phi1(x):
    return (quad(psi1_plus, 0, 1, args=(x))[0] + quad(psi1_minus, -1, 0, args=(x))[0])
def phi2(x):
    return (quad(psi2_plus, 0, 1, args=(x))[0] + quad(psi2_minus, -1, 0, args=(x))[0])
def phi3(x):
    return (quad(psi3_plus, 0, 1, args=(x))[0] + quad(psi3_minus, -1, 0, args=(x))[0])

# Integrands for current
def mu_psi1_plus(mu,x):
    return mu*psi1_plus(mu,x)
def mu_psi2_plus(mu,x):
    return mu*psi2_plus(mu,x)
def mu_psi3_plus(mu,x):
    return mu*psi3_plus(mu,x)
def mu_psi1_minus(mu,x):
    return mu*psi1_minus(mu,x)
def mu_psi2_minus(mu,x):
    return mu*psi2_minus(mu,x)
def mu_psi3_minus(mu,x):
    return mu*psi3_minus(mu,x)

# Current
def J1(x):
    return quad(mu_psi1_plus, 0, 1, args=(x))[0] + quad(mu_psi1_minus, -1, 0, args=(x))[0]
def J2(x):
    return quad(mu_psi2_plus, 0, 1, args=(x))[0] + quad(mu_psi2_minus, -1, 0, args=(x))[0]
def J3(x):
    return quad(mu_psi3_plus, 0, 1, args=(x))[0] + quad(mu_psi3_minus, -1, 0, args=(x))[0]

phi_ref      = np.zeros(60)
J_ref        = np.zeros(60)
phi_face_ref = np.zeros(61)
J_face_ref   = np.zeros(61)

for i in range(20):
    phi_ref[i]      = quad(phi1,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J1,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi1(x[i])
    J_face_ref[i]   = J1(x[i])
for i in range(20,40):
    phi_ref[i]      = quad(phi2,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J2,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi2(x[i])
    J_face_ref[i]   = J2(x[i])
for i in range(40,60):
    phi_ref[i]      = quad(phi3,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J3,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi3(x[i])
    J_face_ref[i]   = J3(x[i])
phi_face_ref[60] = phi3(x[60])
J_face_ref[60]   = J3(x[60])

# =============================================================================
# Plot results
# =============================================================================

# Cases to process and error container
N_hist_list = np.logspace(0,7,15).astype(int)
err_phi = []
err_phi_face = []
err_J = []
err_J_face = []

for N_hist in N_hist_list:
    # Neutron flux
    with h5py.File('output_N=%i.h5'%N_hist, 'r') as f:
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][:]
        phi_face_sd = f['tally/flux-face/sdev'][:]
        J         = f['tally/current/mean'][:,0]/dx
        J_sd      = f['tally/current/sdev'][:,0]/dx
        J_face    = f['tally/current-face/mean'][:,0]
        J_face_sd = f['tally/current-face/sdev'][:,0]

    err_phi.append(np.linalg.norm(phi_ref-phi))
    err_phi_face.append(np.linalg.norm(phi_face_ref-phi_face))
    err_J.append(np.linalg.norm(J_ref-J))
    err_J_face.append(np.linalg.norm(J_face_ref-J_face))

    # Plot
    if N_hist%10 == 0:
        # Solution
        plt.plot(x_mid,phi,'-b',label="MC")
        plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
        plt.plot(x_mid,phi_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Flux')
        plt.ylim([0.06,0.16])
        plt.grid()
        plt.legend()
        plt.title(r'$\bar{\phi}_i$, $N=%i$'%N_hist)
        plt.savefig('phi_N=%i.png'%N_hist,dpi=1200)
        plt.clf()

        plt.plot(x[:],phi_face,'-b',label="MC")
        plt.fill_between(x[:],phi_face-phi_face_sd,phi_face+phi_face_sd,alpha=0.2,color='b')
        plt.plot(x[:],phi_face_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Flux')
        plt.ylim([0.045,0.16])
        plt.grid()
        plt.legend()
        plt.title(r'$\phi(x)$, $N=%i$'%N_hist)
        plt.savefig('phi_face_N=%i.png'%N_hist,dpi=1200)
        plt.clf()

        # Solution
        plt.plot(x_mid,J,'-b',label="MC")
        plt.fill_between(x_mid,J-J_sd,J+J_sd,alpha=0.2,color='b')
        plt.plot(x_mid,J_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Current')
        plt.ylim([-0.03,0.045])
        plt.grid()
        plt.legend()
        plt.title(r'$\bar{J}_i$, $N=%i$'%N_hist)
        plt.savefig('J_N=%i.png'%N_hist,dpi=1200)
        plt.clf()

        plt.plot(x[:],J_face,'-b',label="MC")
        plt.fill_between(x[:],J_face-J_face_sd,J_face+J_face_sd,alpha=0.2,color='b')
        plt.plot(x[:],J_face_ref,'--r',label="ref.")
        plt.xlabel(r'$x$, cm')
        plt.ylabel('Current')
        plt.ylim([-0.03,0.045])
        plt.grid()
        plt.legend()
        plt.title(r'$J(x)$, $N=%i$'%N_hist)
        plt.savefig('J_face_N=%i.png'%N_hist,dpi=1200)
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

        N   = N_hist_list[:len(err_phi_face)]
        fac = err_phi_face[-1]/(1/np.sqrt(N[-1]))
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
        
        N   = N_hist_list[:len(err_J)]
        fac = err_J[-1]/(1/np.sqrt(N[-1]))
        plt.plot(N,err_J,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('Error 2-norm')
        plt.grid()
        plt.legend()
        plt.title(r'$\bar{J}_i$ Convergence')
        plt.savefig('J_convergence.png',dpi=1200)
        plt.clf()

        N   = N_hist_list[:len(err_J_face)]
        fac = err_J_face[-1]/(1/np.sqrt(N[-1]))
        plt.plot(N,err_J_face,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('Error 2-norm')
        plt.grid()
        plt.legend()
        plt.title(r'$J(x)$ Convergence')
        plt.savefig('J_face_convergence.png',dpi=1200)
        plt.clf()
