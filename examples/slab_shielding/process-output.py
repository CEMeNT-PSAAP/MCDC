import h5py
import matplotlib.pyplot as plt

with h5py.File("slab_shielding.h5", "r") as output:
    tally = output["tallies/slab_flux"]
    z = tally["grid/z"][:]
    flux = tally["flux/mean"][:]
    flux_sdev = tally["flux/sdev"][:]

dz = z[1:] - z[:-1]
z_mid = 0.5 * (z[:-1] + z[1:])
flux /= dz
flux_sdev /= dz

figure, axis = plt.subplots()
axis.plot(z_mid, flux, label="Flux")
axis.fill_between(
    z_mid,
    flux - flux_sdev,
    flux + flux_sdev,
    alpha=0.25,
    label="Standard deviation",
)
axis.axvline(2.0, color="black", linestyle="--", label="Material interface")
axis.set_xlabel("z [cm]")
axis.set_ylabel("Flux")
axis.grid()
axis.legend()
figure.tight_layout()
figure.savefig("slab_shielding_flux.png", dpi=150)
