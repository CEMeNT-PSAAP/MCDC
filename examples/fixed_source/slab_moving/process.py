import matplotlib.pyplot as plt
import h5py
import numpy as np

# =============================================================================
# Plot
# =============================================================================

# Load output
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = f["tally/grid/t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]

X, T = np.meshgrid(x_mid, t_mid)
plt.pcolormesh(T.T, X.T, phi.T, cmap="viridis")
plt.plot([0, 5, 10, 10, 15], [2, 2, 5, 1, 1], "r")
plt.xlim([t[0], t[-1]])
plt.ylim([x[0], x[-1]])
plt.ylabel(r"Axial position [mfp]")
plt.xlabel(r"Time [mft]")
plt.show()
