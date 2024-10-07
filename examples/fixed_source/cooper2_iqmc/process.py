import matplotlib.pyplot as plt
import h5py
import numpy as np

# Load iqmc result
with h5py.File("output.h5", "r") as f:
    meshx = f["iqmc/grid/x"][:]
    meshy = f["iqmc/grid/y"][:]
    dx = [meshx[1:] - meshx[:-1]][-1]
    x_mid = 0.5 * (meshx[:-1] + meshx[1:])
    phi = f["iqmc/tally/flux/mean"][:]

    f.close()

# =============================================================================
# Flux Plot
# =============================================================================
X, Y = np.meshgrid(x_mid, x_mid)
Z = np.log10(np.abs(phi / phi.min()))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)", rotation=180)

plt.show()
