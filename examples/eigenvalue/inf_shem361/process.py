import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load("SHEM-361.npz") as data:
    data = np.load("SHEM-361.npz")
    SigmaT = data["SigmaT"]
    nuSigmaF = data["nuSigmaF"]
    SigmaS = data["SigmaS"]
    E = data["E"]

G = len(SigmaT)
E_mid = 0.5 * (E[1:] + E[:-1])
dE = E[1:] - E[:-1]

A = np.dot(np.linalg.inv(np.diag(SigmaT) - SigmaS), nuSigmaF)
w, v = np.linalg.eig(A)
idx = w.argsort()[::-1]
w = w[idx]
v = v[:, idx]
k_exact = w[0]
phi_exact = v[:, 0]
if phi_exact[0] < 0.0:
    phi_exact *= -1
    phi_exact[:] /= np.sum(phi_exact[:])
    phi_exact = phi_exact / dE * E_mid

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    phi_avg = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    k = f["k_cycle"][:]
    k_avg = f["k_mean"][()]
    k_sd = f["k_sdev"][()]

norm = np.sum(phi_avg)
phi_avg = phi_avg / norm / dE * E_mid
phi_sd = phi_sd / norm / dE * E_mid

# Plot
N_iter = len(k)
plt.plot(np.arange(2, N_iter + 1), k[1:], "-b", label="MC")
plt.plot(
    np.arange(2, N_iter + 1), np.ones(N_iter - 1) * k_exact, "--r", label="analytical"
)
plt.plot(np.arange(2, N_iter + 1), np.ones(N_iter - 1) * k_avg, ":g", label="MC-avg")
plt.fill_between(
    np.arange(2, N_iter + 1),
    np.ones(N_iter - 1) * (k_avg - k_sd),
    np.ones(N_iter - 1) * (k_avg + k_sd),
    alpha=0.2,
    color="g",
)
plt.xlabel("Iteration #")
plt.ylabel(r"$k$")
plt.grid()
plt.legend()
plt.show()

plt.step(E_mid, phi_avg, "-b", label="MC", where="mid")
plt.fill_between(
    E_mid, phi_avg - phi_sd, phi_avg + phi_sd, alpha=0.2, color="b", step="mid"
)
plt.step(E_mid, phi_exact, "--r", label="analytical", where="mid")
plt.xscale("log")
plt.xlabel(r"$E$, eV")
plt.ylabel(r"$E\phi(E)$")
plt.grid()
plt.legend()
plt.show()
