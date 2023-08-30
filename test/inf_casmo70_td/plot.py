import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import matplotlib.animation as animation

from reference import reference

output = sys.argv[1]

# Load results
output = sys.argv[1]
with np.load("CASMO-70.npz") as data:
    E = data["E"]
    G = data["G"]
    speed = data["v"]
    E_mid = 0.5 * (E[1:] + E[:-1])
    dE = E[1:] - E[:-1]
with h5py.File(output, "r") as f:
    phi = f["tally/flux-t/mean"][:]
    phi_sd = f["tally/flux-t/sdev"][:]
    t = f["tally/grid/t"][:]
    K = len(t) - 1
phi = phi[:, 1:]
phi_sd = phi_sd[:, 1:]

# Neutron density
n = np.zeros(K)
n_sd = np.zeros(K)
for k in range(K):
    n[k] = np.sum(phi[:, k] / speed)
    n_sd[k] = np.linalg.norm(phi_sd[:, k] / speed)

# Reference solution
phi_ref, n_ref = reference(t)

# Neutron density
phi_ref = phi_ref[1:]
for k in range(K):
    phi_ref[k, :] *= E_mid / dE
    phi[:, k] *= E_mid / dE
    phi_sd[:, k] *= E_mid / dE

# Flux - t
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout(pad=3.0)

ax1.plot(t[1:], n, "-b", label="MC")
no = np.zeros_like(n)
(line,) = ax1.plot(t[1:], no, "ko", fillstyle="none")
ax1.fill_between(t[1:], n - n_sd, n + n_sd, alpha=0.2, color="b")
ax1.plot(t, n_ref, "--r", label="Ref.")
ax1.set_xlabel(r"$t$, s")
ax1.set_ylabel("Density")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_title(r"$n_g(t)$")

ax2.grid()
ax2.set_xlabel(r"$E$, MeV")
ax2.set_ylabel(r"$E\phi(E)$")
ax2.set_title(r"$\phi(E,t)$")
ax2.set_xscale("log")
(line1,) = ax2.step([], [], "-b", where="mid", label="MC")
(line2,) = ax2.step([], [], "--r", where="mid", label="Ref.")
ax2.legend()


def animate(k):
    no[k - 1] = 0.0
    no[k] = n[k]
    line.set_data(t[1:], no)
    line1.set_data(E_mid, phi[:, k])
    ax2.collections.clear()
    ax2.fill_between(
        E_mid,
        phi[:, k] - phi_sd[:, k],
        phi[:, k] + phi_sd[:, k],
        alpha=0.2,
        color="b",
        step="mid",
    )
    line2.set_data(E_mid, phi_ref[k, :G])
    ax2.set_ylim([(phi_ref[k, :G]).min(), (phi_ref[k, :G]).max()])
    ax2.legend()
    return line1, line2


simulation = animation.FuncAnimation(fig, animate, frames=K, interval=100)
simulation.save("inf.gif", savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})
simulation.save("inf.mp4", savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})
plt.show()
