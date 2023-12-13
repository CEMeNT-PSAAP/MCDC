import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

import matplotlib as mpl
import h5py
import numpy as np

# from mpl_toolkits.mplot3d import axes3d, Axes3D


# Load iqmc result
# with h5py.File("output.h5", "r") as f:
#     x = f["iqmc/grid/x"][:]
#     dx = [x[1:] - x[:-1]][-1]
#     x_mid = 0.5 * (x[:-1] + x[1:])

#     phi = f["iqmc/flux"][:]
#     f.close()

# Load iqmc result
with h5py.File("output2.h5", "r") as f:
    meshx = f["iqmc/grid/x"][:]
    meshy = f["iqmc/grid/y"][:]
    dx = [meshx[1:] - meshx[:-1]][-1]
    x_mid = 0.5 * (meshx[:-1] + meshx[1:])
    lowX = meshx[:-1]
    highX = meshx[1:]

    phi = f["iqmc/flux"][:]
    q = f["iqmc/source"][:][0, 0, :, :, 0]
    qdotx = f["iqmc/Q11/x"][:][0, 0, :, :, 0]
    qdoty = f["iqmc/Q11/x"][:][0, 0, :, :, 0]

    f.close()

# =============================================================================
# Flux Plot
# =============================================================================
X, Y = np.meshgrid(x_mid, x_mid)
Z = np.log10(np.abs(phi / phi.min()))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)", rotation=180)

ax.view_init(elev=15, azim=20)
# iqmc_label = mpl.lines.Line2D([0], [0], linestyle="none", c="b", marker="o")
# mc_label = mpl.lines.Line2D([0], [0], linestyle="none", c="r", marker="o")
# ax.legend([iqmc_label, mc_label], ["iQMC", "MC/DC"], numpoints=1)
# plt.tight_layout()
plt.show()

# =============================================================================
# Source Plots
# =============================================================================
X, Y = np.meshgrid(x_mid, x_mid)
Z = np.log10(q)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source", rotation=180)
ax.view_init(elev=15, azim=20)


num_steps = 800
y_mid = x_mid.copy()
Nx = num_steps
Ny = num_steps
x = np.linspace(lowX[0], highX[-1], num=num_steps)
y = np.linspace(lowX[0], highX[-1], num=num_steps)
# Nx = meshx.size
# Ny = meshy.size
# x = meshx
# y = meshy
z1 = np.zeros(shape=(Nx, Ny))
z2 = np.zeros(shape=(Nx, Ny))
z3 = np.zeros(shape=(Nx, Ny))
z4 = np.zeros(shape=(Nx, Ny))
points = np.zeros(shape=(int(Nx * Ny), 2))
count = 0
for i in range(Nx):
    zonex = np.argmax((x[i] > lowX) * (x[i] <= highX))
    midx = x_mid[zonex]
    for j in range(Ny):
        zoney = np.argmax((y[j] > lowX) * (y[j] <= highX))
        midy = y_mid[zoney]
        points[count, :] = [x[i], y[j]]
        count += 1
        z1[i, j] = q[zonex, zoney]
        z2[i, j] = q[zonex, zoney] + qdotx[zonex, zoney] * (x[i] - midx)
        z3[i, j] = q[zonex, zoney] + qdoty[zonex, zoney] * (y[j] - midy)
        z4[i, j] = (
            q[zonex, zoney]
            + qdotx[zonex, zoney] * (x[i] - midx)
            + qdoty[zonex, zoney] * (y[j] - midy)
        )

X, Y = np.meshgrid(x, y)
Z1 = np.log10(z1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, z1, edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source", rotation=180)
ax.view_init(elev=15, azim=20)
plt.show()

Z2 = np.log10(z2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, z2, edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source - Tilted X", rotation=180)
ax.view_init(elev=15, azim=20)
plt.show()

Z3 = np.log10(z3)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, z3, edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source - Tilted Y", rotation=180)
ax.view_init(elev=15, azim=20)
plt.show()

Z4 = np.log10(z4)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, z4, edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source - Tilted X,Y", rotation=180)
ax.view_init(elev=15, azim=20)
plt.show()

# =============================================================================
# Scipy Interpolate plots
# =============================================================================

from scipy.interpolate import RegularGridInterpolator

X, Y = np.meshgrid(x_mid, x_mid)
interp = RegularGridInterpolator(
    (x_mid, x_mid), q, method="linear", bounds_error=False, fill_value=None
)
result = interp(points)
result = np.reshape(result, (Nx, Ny))
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, np.log10(result), edgecolor="b", linewidth=0.5, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"Source")
ax.view_init(elev=15, azim=20)
# plt.xlim((0,10))
# plt.ylim((0,10))
plt.title("Source - Scipy Interp. X, Y, XY")
plt.show()
