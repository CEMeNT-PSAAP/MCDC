import numpy as np
import matplotlib.pyplot as plt
import h5py, sys
from matplotlib import cm
from matplotlib import colors


# =============================================================================
# Plot results
# =============================================================================

# MPACT solution
data = np.loadtxt("mpact_TD4-4.txt")
t_ref = data[:, 0]
phi_ref = data[:, 1]

# Grids
with h5py.File("output_td.h5", "r") as f:
    t = f["tally/grid/t"][:]

# Solution
with h5py.File("output_ic.h5", "r") as f:
    norm = f["IC/fission"][()]
with h5py.File("output_td.h5", "r") as f:
    phi = f["tally/fission/mean"][:]
    phi = phi / norm

plt.figure(figsize=(4, 3))
# plt.plot(t_ref, phi_ref,'r-', label='MPACT')

x = 0.5 * (t[1:] + t[:-1])
y = phi
plt.plot(x, y, "bo", fillstyle="none", label="MC/DC")
plt.xlim([0, 8])
plt.xlabel("Time [s]")
plt.ylabel("Normalized fission rate")
plt.grid()
plt.legend()
plt.savefig("C5G7-TD4-4.png", dpi=1200, bbox_inches="tight", pad_inches=0)
plt.show()
