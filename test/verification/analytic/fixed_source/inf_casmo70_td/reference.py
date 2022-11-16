import numpy as np
from scipy.linalg import expm

def reference(t):
    K = len(t)
    # Load material data
    with np.load('CASMO-70.npz') as data:
        SigmaT   = data['SigmaT']
        SigmaC   = data['SigmaC']
        SigmaS   = data['SigmaS']
        nuSigmaF_p = data['nuSigmaF_p']
        SigmaF   = data['SigmaF']
        nu_p     = data['nu_p']
        nu_d     = data['nu_d']
        chi_p    = data['chi_p']
        chi_d    = data['chi_d']
        G        = data['G']
        J        = data['J']
        E        = data['E']
        v        = data['v']
        lamd     = data['lamd']
    SigmaT += SigmaC*0.28

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
    PHI_init[G-1] = v[-1]

    # Analytical particular solution
    PHI_p = np.dot(np.linalg.inv(-A),Q)

    # Allocate solution
    PHI   = np.zeros([K,G+J])

    # Analytical solution
    for k in range(K):
        # Flux
        PHI_h    = np.dot(expm(AV*t[k]),(PHI_init - PHI_p))
        PHI[k,:] = PHI_h + PHI_p
    phi = PHI[:,:G]

    # Density
    n = np.zeros(len(t))
    for k in range(K):
        n[k] = np.sum(phi[k,:]/v)
    return phi, n
