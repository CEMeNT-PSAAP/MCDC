from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

cases = (
    ("left_20", "20% left / 80% right"),
    ("left_50", "50% left / 50% right"),
    ("left_80", "80% left / 20% right"),
)

profiles = {}
uncertainties = {}
z = None

for case_name, _ in cases:
    output_path = Path(f"source_mix_{case_name}.h5")
    with h5py.File(output_path, "r") as output:
        tally = output["tallies/source_mix_flux"]
        case_z = tally["grid/z"][:]
        flux = tally["flux/mean"][:]
        flux_sdev = tally["flux/sdev"][:]

    if z is None:
        z = case_z
    elif not np.array_equal(z, case_z):
        raise ValueError(f"Inconsistent tally grid in {output_path}")

    dz = case_z[1:] - case_z[:-1]
    profiles[case_name] = flux / dz
    uncertainties[case_name] = flux_sdev / dz

z_mid = 0.5 * (z[:-1] + z[1:])
dz = z[1:] - z[:-1]
left_half = z_mid < 5.0
right_half = ~left_half

print("Integrated flux comparison")
print("--------------------------")
for case_name, label in cases:
    profile = profiles[case_name]
    left_flux = np.sum(profile[left_half] * dz[left_half])
    right_flux = np.sum(profile[right_half] * dz[right_half])
    print(
        f"{label:24s}  left={left_flux:.6e}  right={right_flux:.6e}  "
        f"left/right={left_flux / right_flux:.4f}"
    )

mirror_difference = np.linalg.norm(profiles["left_20"] - profiles["left_80"][::-1])
mirror_scale = np.linalg.norm(0.5 * (profiles["left_20"] + profiles["left_80"][::-1]))
balanced_difference = np.linalg.norm(profiles["left_50"] - profiles["left_50"][::-1])
balanced_scale = np.linalg.norm(profiles["left_50"])

print()
print(
    "20/80 versus mirrored 80/20 relative RMS difference: "
    f"{mirror_difference / mirror_scale:.4e}"
)
print(
    "50/50 profile relative left-right asymmetry: "
    f"{balanced_difference / balanced_scale:.4e}"
)

figure, axis = plt.subplots()
for case_name, label in cases:
    profile = profiles[case_name]
    uncertainty = uncertainties[case_name]
    axis.plot(z_mid, profile, label=label)
    axis.fill_between(
        z_mid,
        profile - uncertainty,
        profile + uncertainty,
        alpha=0.15,
    )

axis.axvline(5.0, color="black", linestyle="--", linewidth=1.0)
axis.set_xlabel("z [cm]")
axis.set_ylabel("Flux")
axis.grid()
axis.legend()
figure.tight_layout()
figure.savefig("iterative_source_comparison.png", dpi=150)
plt.close(figure)
