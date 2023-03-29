import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Plot results
# =============================================================================

# Load iqmc result
with h5py.File("davidson_output.h5", "r") as f:
    x = f["iqmc/grid/x"][:]
    y = f["iqmc/grid/y"][:]
    phi_avg = f["tally/iqmc_flux"][:]
    f.close()


dx = x[1] - x[0]
x_mid = 0.5 * (x[1:] + x[:-1])
y_mid = 0.5 * (y[1:] + y[:-1])
Y, X = np.meshgrid(x_mid, y_mid)

norm = np.sum(phi_avg)
phi_avg = phi_avg.sum(axis=0) / norm

plt.figure(dpi=300)
plt.pcolormesh(X, Y, phi_avg, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Neutron Flux")
plt.show()


Z = np.log10(np.abs(phi_avg / phi_avg.min()))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)")

ax.view_init(elev=15, azim=20)
