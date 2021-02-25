import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation


# =============================================================================
# Reference solution
#   via alpha-eigenfunction expansion
# =============================================================================

# Load XS and grids
with np.load('XS.npz') as data:
    speeds = data['v']        # cm/s
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu     = data['nu']
    E      = data['E']        # eV
with h5py.File('output_N=1.h5', 'r') as f:
    t = f['tally/time_grid'][:]

G     = len(E)-1
K     = len(t)-1
E_mid = 0.5*(E[:-1] + E[1:])
dE    = E[1:] - E[:-1]
dt    = t[1:] - t[:-1]

# Augment with uniform leakage XS
SigmaL  = 0.24 # /cm
SigmaT += SigmaL

# Build the forward and adjoint alpha-eigenvalue operators
nuSigmaF = SigmaF.dot(np.diag(nu))
M     = nuSigmaF + SigmaS - np.diag(SigmaT)
M_adj = np.transpose(M)

# Solve forward eigenvalue problem
w,v = np.linalg.eig(np.diag(speeds).dot(M))
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]

# Solve adjoint eigenvalue problem
w_adj,v_adj = np.linalg.eig(np.diag(speeds).dot(M_adj))
idx = w_adj.argsort()[::-1]   
w_adj = w_adj[idx]
v_adj = v_adj[:,idx]

# Independent source
S = np.zeros(G)
S[-1] = 1.0

# Determine expansion coefficient
A = np.zeros(G)
for n in range(G):
    A[n] = np.dot(v_adj[:,n],S)/np.dot(v_adj[:,n],
                                       np.diag(1/speeds).dot(v[:,n]))

def phi_analytical(t):
    tot = np.zeros(G)
    for n in range(G):
        tot[:] += A[n]*v[:,n]*(np.exp(w[n]*t) - 1.0)/w[n]
    return tot

def phi_analytical_avg(t0,t1):
    dt = t1 - t0
    tot = np.zeros(G)
    for n in range(G):
        tot[:] += A[n]*v[:,n]*((np.exp(w[n]*t1)-np.exp(w[n]*t0))/w[n] - dt)/w[n]
    return tot/dt

# Generate the reference solutions
phi_ref = np.zeros([K,G])
phi_edge_ref = np.zeros([K,G])
for k in range(K):
    t0 = t[k]
    t1 = t[k+1]
    phi_ref[k,:] = phi_analytical_avg(t0,t1)/dE*E_mid
    phi_edge_ref[k,:] = phi_analytical(t[k+1])/dE*E_mid

# =============================================================================
# Plot results
# =============================================================================

# Cases to process and error container
N_hist_list = np.logspace(0,5,11).astype(int)
err_phi = []
err_phi_edge = []

for N_hist in N_hist_list:
    # Neutron flux
    with h5py.File('output_N=%i.h5'%N_hist, 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid*t[-1]
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid*t[-1]
        phi_edge = f['tally/flux-edge/mean'][:]/dE*E_mid*t[-1]
        phi_edge_sd = f['tally/flux-edge/sdev'][:]/dE*E_mid*t[-1]
    
    # Take the time average
    for k in range(K):
        phi[k,:] /= dt[k]
        phi_sd[k,:] /= dt[k]
        
    # Get error
    err_phi.append(np.linalg.norm(phi_ref-phi))
    err_phi_edge.append(np.linalg.norm(phi_edge_ref-phi_edge))
    
    # Plot
    if N_hist%10 == 0:
        fig = plt.figure()
        ax = plt.axes(xlim=(E_mid[0], E_mid[-1]), ylim=(-0.1, 14.0))
        ax.set_xscale('log')    
        ax.set_xlabel(r'$E$, eV')
        ax.set_ylabel(r'$E\bar{\phi}_k(E)$')    
        line1, = ax.plot([], [],'-b',label="MC")
        line2, = ax.plot([], [],'--r',label="reference")
        text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
        ax.legend(loc=6)        
        def animate(k):        
            line1.set_data(E_mid,phi[k,:])
            ax.collections.clear()
            ax.fill_between(E_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
            line2.set_data(E_mid,phi_ref[k,:])
            text.set_text(r'$t_k\in[%.2f,%.2f]$ $\mu s$'%(t[k]*1E6,t[k+1]*1E6))
            return line1, line2, text
        
        simulation = animation.FuncAnimation(fig, animate, frames=K, blit=True)
        writervideo = animation.FFMpegWriter(fps=30)
        simulation.save('phi_N = %i.mp4'%N_hist, writer=writervideo,dpi=300)

        fig = plt.figure()
        ax = plt.axes(xlim=(E_mid[0], E_mid[-1]), ylim=(-0.1, 14.0))
        ax.set_xscale('log')    
        ax.set_xlabel(r'$E$, eV')
        ax.set_ylabel(r'$E\phi(E,t)$')    
        line1, = ax.plot([], [],'-b',label="MC")
        line2, = ax.plot([], [],'--r',label="reference")
        text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
        ax.legend(loc=6)        
        def animate(k):        
            line1.set_data(E_mid,phi_edge[k,:])
            ax.collections.clear()
            ax.fill_between(E_mid,phi_edge[k,:]-phi_edge_sd[k,:],phi_edge[k,:]+phi_edge_sd[k,:],alpha=0.2,color='b')
            line2.set_data(E_mid,phi_edge_ref[k,:])
            text.set_text(r'$t=%.2f$ $\mu s$'%(t[k+1]*1E6))
            return line1, line2, text
        
        simulation = animation.FuncAnimation(fig, animate, frames=K, blit=True)
        writervideo = animation.FFMpegWriter(fps=30)
        simulation.save('phi_edge_N = %i.mp4'%N_hist, writer=writervideo,dpi=300)

        # Convergence
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

        # Convergence
        N   = N_hist_list[:len(err_phi_edge)]
        fac = err_phi_edge[-1]/(1/np.sqrt(N[-1]))
        plt.plot(N,err_phi_edge,'bo',fillstyle='none')
        plt.plot(N,fac/np.sqrt(N),'r-',label=r'$N^{-1/2}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$N$')
        plt.ylabel('Error 2-norm')
        plt.grid()
        plt.legend()
        plt.savefig('phi_edge_convergence.png',dpi=1200)
        plt.clf()