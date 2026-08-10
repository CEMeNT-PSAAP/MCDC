import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("hybrid_multigroup.h5", "r") as output:
    tally = output["tallies/hybrid_flux"]
    z = tally["grid/z"][:]
    energy = tally["grid/energy"][:]
    flux = tally["flux/mean"][:]

dz = z[1:] - z[:-1]
z_midpoint = 0.5 * (z[:-1] + z[1:])
flux = np.reshape(flux, (len(energy) - 1, len(z) - 1)) / dz

figure, axis = plt.subplots()
for index in range(len(energy) - 1):
    axis.plot(
        z_midpoint,
        flux[index],
        marker="o",
        label=f"{energy[index]:.1e}–{energy[index + 1]:.1e} eV",
    )

axis.set_xlabel("z [cm]")
axis.set_ylabel("Flux")
axis.grid()
axis.legend()
figure.tight_layout()
figure.savefig("hybrid_multigroup_flux.png", dpi=150)
