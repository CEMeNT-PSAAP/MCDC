import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from matplotlib import colors


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    phi_avg = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    fis_avg = f["tally/fission/mean"][:]
    fis_sd = f["tally/fission/sdev"][:]
    k = f["k_cycle"][:]
    k_avg = f["k_mean"][()]
    k_sd = f["k_sdev"][()]
    rg = f["gyration_radius"][:]
    x = f["tally/grid/x"][:]
    y = f["tally/grid/y"][:]
    z = f["tally/grid/z"][:]

dx = x[1] - x[0]
dz = z[1] - z[0]
dV = dx * dx * dz

norm = np.sum(phi_avg)
phi_avg = phi_avg / norm / dV
phi_sd = phi_sd / norm / dV
fis_avg = fis_avg / norm / dV
fis_sd = fis_sd / norm / dV

phi_fast = np.zeros_like(phi_avg[0])
phi_thermal = np.zeros_like(phi_avg[0])
phi_fast_sd = np.zeros_like(phi_avg[0])
phi_thermal_sd = np.zeros_like(phi_avg[0])
fis_ = np.zeros_like(phi_avg[0])
fis__sd = np.zeros_like(phi_avg[0])

for i in range(2):
    phi_fast += phi_avg[i]
    phi_fast_sd += np.square(phi_sd[i])
phi_fast_sd = np.sqrt(phi_fast_sd)
for i in range(2, 7):
    phi_thermal += phi_avg[i]
    phi_thermal_sd += np.square(phi_sd[i])
phi_thermal_sd = np.sqrt(phi_thermal_sd)
for i in range(7):
    fis_ += fis_avg[i]
    fis__sd += np.square(fis_sd[i])
fis__sd = np.sqrt(fis__sd)

print("k = %.5f +- %.5f" % (k_avg, k_sd))

# Plot
N_iter = len(k)
(p1,) = plt.plot(np.arange(1, N_iter + 1), k, "-b", label="MC")
(p2,) = plt.plot(
    np.arange(1, N_iter + 1), np.ones(N_iter) * k_avg, ":r", label="MC-avg"
)
plt.fill_between(
    np.arange(1, N_iter + 1),
    np.ones(N_iter) * (k_avg - k_sd),
    np.ones(N_iter) * (k_avg + k_sd),
    alpha=0.2,
    color="r",
)
plt.xlabel("Iteration #")
plt.ylabel(r"$k$")
plt.grid()
ax2 = plt.gca().twinx()
(p3,) = ax2.plot(np.arange(1, N_iter + 1), rg, "g--", label="GyRad")
plt.ylabel(r"Gyration radius [cm]")
lines = [p1, p2, p3]
plt.legend(lines, [l.get_label() for l in lines])
plt.show()

# X-Z plane
x_mid = 0.5 * (x[1:] + x[:-1])
z_mid = 0.5 * (z[1:] + z[:-1])
X, Z = np.meshgrid(z_mid, x_mid)
phi_fast_xz = np.sum(phi_fast, axis=1)
phi_thermal_xz = np.sum(phi_thermal, axis=1)
fis_xz = np.sum(fis_, axis=1)
phi_fast_xz_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=1)) / phi_fast_xz
phi_thermal_xz_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=1)) / phi_thermal_xz
fis_xz_sd = np.sqrt(np.sum(np.square(fis__sd), axis=1)) / fis_xz

plt.pcolormesh(Z, X, phi_fast_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_fast_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()

plt.pcolormesh(Z, X, fis_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fission rate")
plt.show()

plt.pcolormesh(Z, X, fis_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fission rate stdev")
plt.show()

# X-Y plane
Y, X = np.meshgrid(x_mid, x_mid)
phi_fast_xy = np.sum(phi_fast, axis=2)
phi_thermal_xy = np.sum(phi_thermal, axis=2)
fis_xy = np.sum(fis_, axis=2)
phi_fast_xy_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=2)) / phi_fast_xy
phi_thermal_xy_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=2)) / phi_thermal_xy
fis_xy_sd = np.sqrt(np.sum(np.square(fis__sd), axis=2)) / fis_xy

plt.pcolormesh(X, Y, phi_fast_xy, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_fast_xy_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(X, Y, phi_thermal_xy, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_thermal_xy_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()

plt.pcolormesh(X, Y, fis_xy, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fission rate")
plt.show()

plt.pcolormesh(X, Y, fis_xy_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fission rate stdev")
plt.show()


"""
# Mesh grid
X,Y,Z = np.meshgrid(x,y,z)

# Voxel fill
fill = np.zeros_like(phi_fast, dtype=bool)
Nx = len(x)-1
Ny = len(y)-1
Nz = len(z)-1

fill[int(Nx/2):,int(Ny/2):,:] = True

# Convert data to colors
norm = colors.Normalize()
phi_fast = cm.viridis(norm(phi_fast))

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(X, Y, Z, fill, facecolors=phi_fast)
ax.set(xlabel='x (cm)', ylabel='y (cm)', zlabel='z (cm)')
plt.show()
"""
