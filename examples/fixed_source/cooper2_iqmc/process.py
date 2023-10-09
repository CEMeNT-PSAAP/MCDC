import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

import matplotlib as mpl
import h5py
import numpy as np


# Load iqmc result
with h5py.File("output.h5", "r") as f:
    x = f["iqmc/grid/x"][:]
    dx = [x[1:] - x[:-1]][-1]
    x_mid = 0.5 * (x[:-1] + x[1:])

    phi = f["tally/iqmc_flux"][:]
    f.close()
# compare with MCDC result
# with h5py.File("..\cooper2\output.h5", "r") as f:
#     x = f["tally/grid/x"][:]
#     dx = [x[1:] - x[:-1]][-1]
#     x_mid = 0.5 * (x[:-1] + x[1:])

#     phi_mc = f["tally/flux/mean"][:]
#     phi_sd = f["tally/flux/sdev"][:]

#     f.close()

# Plot result
X, Y = np.meshgrid(x_mid, x_mid)

Z = np.log10(np.abs(phi / phi.min()))
# Z2 = np.log10(np.abs(phi_mc) / (dx**2))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)
# ax.plot_surface(Y, X, Z2, edgecolor="r", color="white", linewidth=0.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)", rotation=180)

ax.view_init(elev=15, azim=20)

# iqmc_label = mpl.lines.Line2D([0], [0], linestyle="none", c="b", marker="o")
# mc_label = mpl.lines.Line2D([0], [0], linestyle="none", c="r", marker="o")

# ax.legend([iqmc_label, mc_label], ["iQMC", "MC/DC"], numpoints=1)

# plt.tight_layout()
plt.show()
