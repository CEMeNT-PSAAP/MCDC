import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys, openmc
import matplotlib.animation as animation


# Grids
with np.load("SHEM-361.npz") as data:
    E = data["E"]
    G = data["G"]
    speed = data["v"]
    E_mid = 0.5 * (E[1:] + E[:-1])
    dE = E[1:] - E[:-1]
t = np.insert(np.logspace(-8, 2, 50), 0, 0.0)
dt = t[1:] - t[:-1]
t_mid = 0.5 * (t[1:] + t[:-1])
K = len(t) - 1

# OpenMC results
with openmc.StatePoint('statepoint.100.h5') as sp:
    tally = sp.get_tally(name="TD spectrum")
    phi_openmc = (tally.get_values(scores=['flux'])).reshape(K,G)
    phi_sd_openmc = (tally.get_values(scores=['flux'],value='std_dev')).reshape(K,G)

# MCDC
with h5py.File('output.h5', "r") as f:
    phi_mcdc = f["tally/flux/mean"][:]
    phi_sd_mcdc = f["tally/flux/sdev"][:]

# Neutron density
n_openmc = np.zeros(K)
n_sd_openmc = np.zeros(K)
n_mcdc = np.zeros(K)
n_sd_mcdc = np.zeros(K)
for k in range(K):
    n_openmc[k] = np.sum(phi_openmc[k, :] / speed) / dt[k]
    n_sd_openmc[k] = np.linalg.norm(phi_sd_openmc[k, :] / speed) / dt[k]
    n_mcdc[k] = np.sum(phi_mcdc[:, k] / speed) / dt[k]
    n_sd_mcdc[k] = np.linalg.norm(phi_sd_mcdc[:, k] / speed) / dt[k]

# Normalize
for k in range(K):
    phi_openmc[k, :] *= E_mid / dE / dt[k]
    phi_sd_openmc[k,:] *= E_mid / dE / dt[k]
    phi_mcdc[:, k] *= E_mid / dE / dt[k]
    phi_sd_mcdc[:, k] *= E_mid / dE / dt[k]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout(pad=3.0)

ax1.plot(t_mid, n_openmc, "-b", label="OpenMC")
ax1.plot(t_mid, n_mcdc, "--r", label="MC/DC")
no_openmc = np.zeros_like(n_openmc)
no_mcdc = np.zeros_like(n_mcdc)
(dot_openmc,) = ax1.plot(t[1:], no_openmc, "bo", fillstyle="none")
(dot_mcdc,) = ax1.plot(t[1:], no_mcdc, "ro", fillstyle="none")
y = n_openmc
y_sd = n_sd_openmc
ax1.fill_between(t[1:], y - y_sd, y + y_sd, alpha=0.2, color="b")
y = n_mcdc
y_sd = n_sd_mcdc
ax1.fill_between(t[1:], y - y_sd, y + y_sd, alpha=0.2, color="r")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("Density")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_title("Neutron density")

ax2.grid()
ax2.set_xlabel("Energy [eV]")
ax2.set_ylabel(r"Spectrum, $E\phi(E)$")
ax2.set_title(r"Flux energy spectrum")
ax2.set_xscale("log")
ax2.set_xlim((E_mid[0],E_mid[-1]))
(line_openmc,) = ax2.plot([], [], "-b", label="OpenMC")
(line_mcdc,) = ax2.plot([], [], "--r", label="MC/DC")
fb_openmc = ax2.fill_between([], [], [], [], alpha=0.2, color="b")
fb_mcdc = ax2.fill_between([], [], [], [], alpha=0.2, color="r")
ax2.legend()


def animate(k):
    global fb_openmc, fb_mcdc
    fb_openmc.remove()
    fb_mcdc.remove()
    no_openmc[k - 1] = 0.0
    no_openmc[k] = n_openmc[k]
    no_mcdc[k - 1] = 0.0
    no_mcdc[k] = n_mcdc[k]
    dot_openmc.set_data(t[1:], no_openmc)
    dot_mcdc.set_data(t[1:], no_mcdc)
    line_openmc.set_data(E_mid, phi_openmc[k,:])
    line_mcdc.set_data(E_mid, phi_mcdc[:,k])
    y = phi_openmc[k,:]
    y_sd = phi_sd_openmc[k,:]
    fb_openmc = ax2.fill_between(
        E_mid,
        y-y_sd,
        y+y_sd,
        alpha=0.2,
        color="b",
    )
    y = phi_mcdc[:,k]
    y_sd = phi_sd_mcdc[:,k]
    fb_mcdc = ax2.fill_between(
        E_mid,
        y-y_sd,
        y+y_sd,
        alpha=0.2,
        color="r",
    )
    ax2.legend()
    ax2.set_ylim([min((phi_openmc[k,:]).min(), (phi_mcdc[:,k]).min()), max((phi_openmc[k,:]).max(), (phi_mcdc[:,k]).max())])
    return line_openmc, line_mcdc


simulation = animation.FuncAnimation(fig, animate, frames=K)
#writervideo = animation.FFMpegWriter(fps=10)
simulation.save('pin_1MeV.gif',fps=10,writer='imagemagick',  savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0}, dpi=600)
plt.show()
