import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from matplotlib import colors


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("dd_1e3.h5", "r") as f:
    phi_avg = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    x = f["tally/grid/x"][:]
    y = f["tally/grid/y"][:]
    z = f["tally/grid/z"][:]

with h5py.File("ref1e3.h5", "r") as f:
    phi_avgr = f["tally/flux/mean"][:]
    phi_sdr = f["tally/flux/sdev"][:]

diff = np.abs(phi_avgr-phi_avg)
diff = np.sum(diff)/diff.size
print("total difference:",diff,"sd",np.average(phi_sdr))

dx = x[1] - x[0]
dz = z[1] - z[0]
dV = dx * dx * dz

phi_avgr /= dV
phi_sdr /= dV

phi_fastr = phi_avgr[0]
phi_thermalr = phi_avgr[1]
phi_fast_sdr = phi_sdr[0]
phi_thermal_sdr = phi_sdr[1]

# X-Y plane
phi_fast_xyr = np.sum(phi_fastr, axis=2)
phi_thermal_xyr = np.sum(phi_thermalr, axis=2)
phi_fast_xy_sdr = np.sqrt(np.sum(np.square(phi_fast_sdr), axis=2)) / phi_fast_xyr
phi_thermal_xy_sdr = np.sqrt(np.sum(np.square(phi_thermal_sdr), axis=2)) / phi_thermal_xyr

# X-Z plane
phi_fast_xzr = np.sum(phi_fastr, axis=1)
phi_thermal_xzr = np.sum(phi_thermalr, axis=1)
phi_fast_xz_sdr = np.sqrt(np.sum(np.square(phi_fast_sdr), axis=1)) / phi_fast_xzr
phi_thermal_xz_sdr = np.sqrt(np.sum(np.square(phi_thermal_sdr), axis=1)) / phi_thermal_xzr

phi_avg /= dV
phi_sd /= dV

phi_fast = phi_avg[0]
phi_thermal = phi_avg[1]
phi_fast_sd = phi_sd[0]
phi_thermal_sd = phi_sd[1]

# X-Y plane
x_mid = 0.5 * (x[1:] + x[:-1])
Y, X = np.meshgrid(x_mid, x_mid)


phi_fast_xy = np.sum(phi_fast, axis=2)
phi_thermal_xy = np.sum(phi_thermal, axis=2)
phi_fast_xy_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=2)) / phi_fast_xy
phi_thermal_xy_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=2)) / phi_thermal_xy

phi_fast_xz = np.sum(phi_fast, axis=1)
phi_thermal_xz = np.sum(phi_thermal, axis=1)
phi_fast_xz_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=1)) / phi_fast_xz
phi_thermal_xz_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=1)) / phi_thermal_xz

plt.clf()
fig, axes = plt.subplots(1,3,figsize=(18,6))

mesh_flux = axes[0].pcolormesh(X, Y, phi_thermal_xyr, shading="nearest")
axes[0].set_title("Analog")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

mesh_fluxr = axes[1].pcolormesh(X, Y, phi_thermal_xy, shading="nearest")
axes[1].set_title("Domain Decomp")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

mesh_diff = axes[2].pcolormesh(X, Y, np.abs(phi_thermal_xy-phi_thermal_xyr), shading="nearest")
axes[2].set_title("Difference")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

fig.colorbar(mesh_flux, ax=axes[0])
fig.colorbar(mesh_fluxr, ax=axes[1])
fig.colorbar(mesh_diff, ax=axes[2])
plt.savefig("flux_comparison_thermal_xy.png")
z_mid = 0.5 * (z[1:] + z[:-1])
X, Z = np.meshgrid(z_mid, x_mid)
plt.clf()
fig, axes = plt.subplots(1,3,figsize=(18,6))

mesh_flux = axes[0].pcolormesh(Z, X, phi_thermal_xzr, shading="nearest")
axes[0].set_title("Analog")
axes[0].set_xlabel("x")
axes[0].set_ylabel("z")

mesh_fluxr = axes[1].pcolormesh(Z, X, phi_thermal_xz, shading="nearest")
axes[1].set_title("Domain Decomp")
axes[0].set_xlabel("x")
axes[0].set_ylabel("z")

mesh_diff = axes[2].pcolormesh(Z,X, np.abs(phi_thermal_xz-phi_thermal_xzr), shading="nearest")
axes[2].set_title("Difference")
axes[2].set_xlabel("x")
axes[2].set_ylabel("z")

fig.colorbar(mesh_flux, ax=axes[0])
fig.colorbar(mesh_fluxr, ax=axes[1])
fig.colorbar(mesh_diff, ax=axes[2])
plt.savefig("flux_comparison_thermal_xz.png")

