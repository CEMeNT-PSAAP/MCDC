import numpy as np
from scipy.linalg import expm
from scipy.integrate import trapz

# Time grid
t = np.insert(np.logspace(-8, 1, 100), 0, 0.0)
K = len(t) - 1

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaT = data["SigmaT"]
    SigmaC = data["SigmaC"]
    SigmaS = data["SigmaS"]
    nuSigmaF_p = data["nuSigmaF_p"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"]
    J = data["J"]
    E = data["E"]
    v = data["v"]
    lamd = data["lamd"]
SigmaT += SigmaC * 0.28

# Matrix and RHS source
A = np.zeros([G + J, G + J])

# Top-left [GxG]: phi --> phi
A[:G, :G] = SigmaS + nuSigmaF_p - np.diag(SigmaT)

# Top-right [GxJ]: C --> phi
A[:G, G:] = np.multiply(chi_d, lamd)

# Bottom-left [JxG]: phi --> C
A[G:, :G] = np.multiply(nu_d, SigmaF)

# bottom-right [JxJ]: C --> C
A[G:, G:] = -np.diag(lamd)

# Multiply with neutron speed
AV = np.copy(A)
AV[:G, :] = np.dot(np.diag(v), A[:G, :])

# Initial condition
PHI_init = np.zeros(G + J)
PHI_init[G - 1] = v[-1]

# Allocate solution
PHI = np.zeros([K, G + J])

# Analytical solution
PHI_OLD = PHI_init
for k in range(K):
    print(k)
    t1 = t[k]
    t2 = t[k + 1]
    dt = t2 - t1

    # Fluxes
    PHI_NEW = np.dot(expm(AV * dt), PHI_OLD)
    PHI[k, :] = np.dot(np.linalg.inv(AV), (PHI_NEW - PHI_OLD) / dt)
    PHI_OLD[:] = PHI_NEW[:]
phi = PHI[:, :G]

# Density
n = np.zeros(K)
for k in range(K):
    n[k] = np.sum(phi[k, :] / v)

np.savez("reference.npz", t=t, phi=phi, n=n)
