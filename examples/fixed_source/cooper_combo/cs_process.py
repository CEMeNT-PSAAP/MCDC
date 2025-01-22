import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import cvxpy as cp

with h5py.File("output.h5", "r") as f:
    # User-defined parameters
    x_grid = np.linspace(0.0, 4.0, 41)
    y_grid = np.linspace(0.0, 4.0, 41)

    # Parameters not defined by the user
    Nx = len(x_grid) - 1
    Ny = len(y_grid) - 1
    N_cs_bins = f["tallies"]["cs_tally_0"]["N_cs_bins"][()]
    cs_bin_size = f["tallies"]["cs_tally_0"]["cs_bin_size"]
    x_centers = f["tallies"]["cs_tally_0"]["center_points"][0]
    y_centers = f["tallies"]["cs_tally_0"]["center_points"][1]
    x_centers[-1] = (x_grid[-1] + x_grid[0]) / 2
    y_centers[-1] = (y_grid[-1] + y_grid[0]) / 2

    # Construct S
    S = [[] for _ in range(N_cs_bins)]
    # Calculate the overlap grid for each bin, and flatten into a row of S
    for ibin in range(N_cs_bins):
        if ibin == N_cs_bins - 1:
            # could just change to -INF, INF ?
            cs_bin_size = np.array([x_grid[-1] + x_grid[0], y_grid[-1] + y_grid[0]])

        bin_x_min = x_centers[ibin] - cs_bin_size[0] / 2
        bin_x_max = x_centers[ibin] + cs_bin_size[0] / 2
        bin_y_min = y_centers[ibin] - cs_bin_size[1] / 2
        bin_y_max = y_centers[ibin] + cs_bin_size[1] / 2

        overlap = np.zeros((len(y_grid) - 1, len(x_grid) - 1))

        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                cell_x_min = x_grid[j]
                cell_x_max = x_grid[j + 1]
                cell_y_min = y_grid[i]
                cell_y_max = y_grid[i + 1]

                # Calculate overlap in x and y directions
                overlap_x = np.maximum(
                    0,
                    np.minimum(bin_x_max, cell_x_max)
                    - np.maximum(bin_x_min, cell_x_min),
                )
                overlap_y = np.maximum(
                    0,
                    np.minimum(bin_y_max, cell_y_max)
                    - np.maximum(bin_y_min, cell_y_min),
                )

                # Calculate fractional overlap
                cell_area = (cell_x_max - cell_x_min) * (cell_y_max - cell_y_min)
                overlap[i, j] = (overlap_x * overlap_y) / cell_area

        S[ibin] = overlap.flatten()
    S = np.array(S)

    cs_results = f["tallies"]["cs_tally_0"]["flux"]["mean"][:]
    mesh_results = f["tallies"]["mesh_tally_0"]["flux"]["mean"][:]

# Perform reconstruction
N_fine_cells = Nx * Ny
b = cs_results  # measurement vector for cs reconstruction

# Constructing T and A
idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

T_inv = np.kron(idct_basis_y, idct_basis_x)
A = S @ T_inv


# "l" is short for lambda, which controls the amount of sparsity vs reconstruction accuracy
def reconstruct(l):
    # Basis pursuit denoising solver - change l to get different results
    vx = cp.Variable(N_fine_cells)
    objective = cp.Minimize(0.5 * cp.norm(A @ vx - b, 2) + l * cp.norm(vx, 1))
    constraints = [(A @ vx)[-1] == b[-1]]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    sparse_solution = np.array(vx.value).squeeze()

    # Obtaining the reconstruction
    recon = T_inv @ sparse_solution
    recon_reshaped = recon.reshape(Ny, Nx)
    return recon_reshaped


def rel_norm(real, recon):
    real_norm = np.linalg.norm(real.flatten(), ord=2)
    diff = real.flatten() - recon.flatten()
    return np.linalg.norm(diff) / real_norm


# Different values of lambda to reconstruct with
l_array = [
    0,
    0.00001,
    0.00005,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1,
    10,
]

# This part plots 16 different reconstructions for the given values of lambda in l_array
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        im = ax.imshow(
            reconstruct(l_array[i * 4 + j]), origin="lower", extent=[0, 4, 0, 4]
        )
        ax.set_title(f"$\lambda$ = {l_array[i * 4 + j]:.5g}")

        if j == 0:
            ax.set_ylabel("y [cm]")
        if i == 3:
            ax.set_xlabel("x [cm]")

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=1)
        cbar.formatter.set_powerlimits((0, 0))

plt.suptitle(
    "Basis Pursuit Denoising Reconstructions with Different Values of $\lambda$",
    fontsize=16,
)
plt.tight_layout()
# plt.savefig('BPDN_lambda_testing_sphere.png')
plt.show()

rel_norms = np.ones(len(l_array))
for i in range(len(l_array)):
    rel_norms[i] = rel_norm(reconstruct(l_array[i]), mesh_results)

reference_std_dev = np.linalg.norm(np.std(mesh_results))
plt.plot(l_array, rel_norms, label="Reconstruction Errors")
plt.hlines(
    reference_std_dev,
    l_array[0],
    l_array[-1],
    color="black",
    linestyle="--",
    label="Reference Std Dev",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("Relative L$^2$ Error")
plt.title("Relative Error vs $\lambda$ - Modified Cooper Reconstructions")
plt.legend()
plt.tight_layout()
# plt.savefig('Relative_error_vs_lambda_sphere.png')
plt.show()
