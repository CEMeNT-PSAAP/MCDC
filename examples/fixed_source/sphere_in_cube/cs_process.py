import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import time

try:
    import cvxpy as cp
except ImportError:
    print("CVXPY has not been installed. Please install it with 'pip install cvxpy'")

# User-defined parameters - number of cells in each dimension
Nx = 20
Ny = 20
Nz = 20

with h5py.File("output.h5", "r") as f:
    x_grid = np.linspace(0.0, 4.0, Nx + 1)
    y_grid = np.linspace(0.0, 4.0, Ny + 1)
    z_grid = np.linspace(0.0, 4.0, Nz + 1)
    N_cs_bins = f["tallies"]["cs_tally_0"]["N_cs_bins"][()]
    cs_bin_size = f["tallies"]["cs_tally_0"]["cs_bin_size"]
    x_centers = f["tallies"]["cs_tally_0"]["center_points"][0]
    y_centers = f["tallies"]["cs_tally_0"]["center_points"][1]
    z_centers = f["tallies"]["cs_tally_0"]["center_points"][2]
    x_centers[-1] = (x_grid[-1] + x_grid[0]) / 2
    y_centers[-1] = (y_grid[-1] + y_grid[0]) / 2
    z_centers[-1] = (z_grid[-1] + z_grid[0]) / 2
    x_mins, x_maxs = x_grid[:-1], x_grid[1:]
    y_mins, y_maxs = y_grid[:-1], y_grid[1:]
    z_mins, z_maxs = z_grid[:-1], z_grid[1:]

    x_mids = (x_mins + x_maxs) / 2
    y_mids = (y_mins + y_maxs) / 2
    z_mids = (z_mins + z_maxs) / 2

    # volume of a single cell
    cell_volumes = np.multiply.outer(
        np.multiply.outer(x_maxs - x_mins, y_maxs - y_mins), z_maxs - z_mins
    )

    # initialize S
    S = np.zeros((N_cs_bins, Nx * Ny * Nz))

    print("Generating S...")
    for ibin in range(N_cs_bins):
        bin_x_min = x_centers[ibin] - cs_bin_size[0] / 2
        bin_x_max = x_centers[ibin] + cs_bin_size[0] / 2
        bin_y_min = y_centers[ibin] - cs_bin_size[1] / 2
        bin_y_max = y_centers[ibin] + cs_bin_size[1] / 2
        bin_z_min = z_centers[ibin] - cs_bin_size[2] / 2
        bin_z_max = z_centers[ibin] + cs_bin_size[2] / 2

        # calculate overlap
        overlap_x = np.maximum(
            0,
            np.minimum(bin_x_max, x_maxs[:, None, None])
            - np.maximum(bin_x_min, x_mins[:, None, None]),
        )
        overlap_y = np.maximum(
            0,
            np.minimum(bin_y_max, y_maxs[None, :, None])
            - np.maximum(bin_y_min, y_mins[None, :, None]),
        )
        overlap_z = np.maximum(
            0,
            np.minimum(bin_z_max, z_maxs[None, None, :])
            - np.maximum(bin_z_min, z_mins[None, None, :]),
        )

        # calculate fractional overlap
        overlap = (overlap_x * overlap_y * overlap_z) / cell_volumes
        S[ibin] = overlap.flatten()

        for i in range(len(S[-1])):
            S[-1][i] = 1

    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]
    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    mesh_sdev = f["tallies"]["mesh_tally_0"]["fission"]["sdev"][:]

# Perform reconstruction
N_fine_cells = Nx * Ny * Nz
b = cs_results  # measurement vector
A = (spfft.dct(S.T, type=2, norm="ortho", axis=0)).T  # sensing matrix

# idct_basis_x = spfft.idct(np.identity(Nx), norm='ortho', axis=0)
# idct_basis_y = spfft.idct(np.identity(Ny), norm='ortho', axis=0)
# T_inv = np.kron(idct_basis_y, idct_basis_x)
# A = S @ T_inv


def reconstruct(lambda_):
    print(f"Reconstructing with lambda = {lambda_}", end="\r")
    start_time = time.time()

    # setting up the problem with CVXPY
    vx = cp.Variable(N_fine_cells)

    # Basis pursuit denoising (BPDN)
    objective = cp.Minimize(
        0.5 * cp.norm(A @ vx - b, 2) ** 2 + lambda_ * cp.norm(vx, 1)
    )
    constraints = [(A @ vx)[-1] == b[-1]]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    # formatting the sparse solution
    sparse_solution = np.array(vx.value).squeeze()
    result = spfft.idct(sparse_solution, type=2, norm="ortho", axis=0)
    recon = result.reshape(Nz, Ny, Nx)
    print(
        f"Reconstructing with lambda = {lambda_}, time = {np.round(time.time() - start_time, 4)}"
    )
    return recon


def rel_norm(recon, real):
    recon = recon.flatten()
    real = real.flatten()
    recon_norm = np.linalg.norm(recon, ord=2)
    real_norm = np.linalg.norm(real, ord=2)

    return np.abs((recon_norm - real_norm) / real_norm)


# Different values of lambda to reconstruct with
l_array = [
    "mesh",
    0,
    0.00005,
    0.0001,
    0.0005,
    0.00075,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.0125,
    0.015,
    0.0175,
    0.02,
    0.03,
]

# Create reconstructions and compute relative norms
recon_array = [mesh_results if l == "mesh" else reconstruct(l) for l in l_array]
rel_norms = [rel_norm(recon, mesh_results) for recon in recon_array]

# Plotting the reconstructions for different lambda values
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    l = l_array[i]
    reconstruction = recon_array[i]
    im = ax.imshow(reconstruction[:, :, Nz // 2], origin="lower", extent=[0, 4, 0, 4])

    ax.set_title(f"True Solution" if l == "mesh" else f"$\lambda$ = {l:.5g}")
    ax.set_ylabel("y [cm]" if i % 4 == 0 else "")
    ax.set_xlabel("x [cm]" if i >= 12 else "")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=1)
    cbar.formatter.set_powerlimits((0, 0))

plt.suptitle(
    "Basis Pursuit Denoising Reconstructions with Different Values of $\lambda$",
    fontsize=16,
)
plt.tight_layout()
plt.savefig("3D Reconstructions of Fissile Sphere.png")
plt.show()

# Plotting the relative errors
plt.plot(l_array[2:], rel_norms[2:], label="Reconstruction Errors")
plt.hlines(
    np.linalg.norm(mesh_sdev.flatten(), ord=2)
    / np.linalg.norm(mesh_results.flatten(), ord=2),
    plt.xlim()[0],
    plt.xlim()[1],
    color="black",
    linestyle="--",
    label="Reference Std Dev",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("Relative L$^2$ Error")
plt.title("Relative Error vs $\lambda$ - Fissile Sphere Reconstructions")
plt.legend()
plt.tight_layout()
plt.savefig("Reconstruction Errors.png")
plt.show()
