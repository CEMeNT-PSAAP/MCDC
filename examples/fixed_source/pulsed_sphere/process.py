import matplotlib.pyplot as plt
import h5py

# Get results
with h5py.File("output.h5", "r") as f:
    t = f["tallies/mesh_tally_0/grid/t"][:]
    dt = t[1:] - t[:-1]
    t_mid = 0.5 * (t[1:] + t[:-1])
    K = len(t) - 1

    phi = f["tallies/mesh_tally_0/flux/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dt[k]
        phi_sd[k] /= dt[k]

plt.plot(t_mid, phi)
plt.yscale("log")
plt.show()
