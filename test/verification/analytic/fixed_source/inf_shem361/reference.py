import numpy as np
from scipy.integrate import quad


def reference():
    # Load material data
    with np.load("SHEM-361.npz") as data:
        SigmaT = data["SigmaT"]
        SigmaC = data["SigmaC"]
        SigmaS = data["SigmaS"]
        nuSigmaF = data["nuSigmaF"]
        G = data["G"]
    SigmaT += SigmaC * 0.5

    A = np.diag(SigmaT) - SigmaS - nuSigmaF
    Q = np.zeros(G)
    Q[-1] = 1.0

    phi = np.linalg.solve(A, Q)
    return phi
