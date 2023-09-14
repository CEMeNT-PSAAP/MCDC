import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import matplotlib.animation as animation

# Reference solution
data = np.load("reference.npz")
phi_ref = data["phi"].T
n_ref = data["n"]

# Load results
output = sys.argv[1]
with np.load("SHEM-361.npz") as data:
    E = data["E"]
    G = data["G"]
    speed = data["v"]
    E_mid = 0.5 * (E[1:] + E[:-1])
    dE = E[1:] - E[:-1]
with h5py.File(output, "r") as f:
    t = f["tally/grid/t"][:]
    dt = t[1:] - t[:-1]
    t_mid = 0.5 * (t[1:] + t[:-1])
    K = len(t) - 1

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]

# Neutron density
n = np.zeros(K)
n_sd = np.zeros(K)
for k in range(K):
    n[k] = np.sum(phi[:, k] / speed) / dt[k]
    n_sd[k] = np.linalg.norm(phi_sd[:, k] / speed) / dt[k]

# Normalize
for k in range(K):
    phi_ref[:, k] *= E_mid / dE
    phi[:, k] *= E_mid / dE / dt[k]
    phi_sd[:, k] *= E_mid / dE / dt[k]

# Flux - t
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout(pad=3.0)

ax1.plot(t_mid, n, "-b", label="MC")
no = np.zeros_like(n)
(line,) = ax1.plot(t[1:], no, "ko", fillstyle="none")
ax1.fill_between(t[1:], n - n_sd, n + n_sd, alpha=0.2, color="b")
ax1.plot(t_mid, n_ref, "--r", label="Ref.")
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
(line1,) = ax2.plot([], [], "-b", label="MC")
(line2,) = ax2.plot([], [], "--r", label="Ref.")
fb = ax2.fill_between([], [], [], [], alpha=0.2, color="b")
ax2.legend()


def animate(k):
    global fb
    fb.remove()
    no[k - 1] = 0.0
    no[k] = n[k]
    line.set_data(t[1:], no)
    line1.set_data(E_mid, phi[:, k])
    fb = ax2.fill_between(
        E_mid,
        phi[:, k] - phi_sd[:, k],
        phi[:, k] + phi_sd[:, k],
        alpha=0.2,
        color="b",
    )
    line2.set_data(E_mid, phi_ref[:, k])
    ax2.set_ylim([(phi_ref[:, k]).min(), (phi_ref[:, k]).max()])
    ax2.legend()
    return line1, line2


simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
