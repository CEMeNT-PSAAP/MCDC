import matplotlib.pyplot as plt
import h5py
import sys

from reference import reference


# Load results
output = sys.argv[1]
with h5py.File(output, "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    phi = f["tally/flux/mean"][:] / dx * 101.0
    phi_sd = f["tally/flux/sdev"][:] / dx * 101.0

# Reference solution
phi_ref = reference(x)

# Flux - spatial average
plt.plot(x_mid, phi, "-b", label="MC")
plt.fill_between(x_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
plt.plot(x_mid, phi_ref, "--r", label="Ref.")
plt.xlabel(r"$x$, cm")
plt.ylabel("Flux")
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()
