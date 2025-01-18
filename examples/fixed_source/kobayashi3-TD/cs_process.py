import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import matplotlib.animation as animation
import scipy.fft as spfft
import cvxpy as cp

with h5py.File("output.h5", "r") as f:
    S = f["tallies"]["cs_tally_0"]["S"][:]
    recon = f["tallies"]["cs_tally_0"]["flux"]["reconstruction"]
    plt.imshow(recon, extent=[0, 60, 100, 0])
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.gca().invert_yaxis()
    plt.title(
        "Reconstruction, $\lambda$ = 0.5"
    )  # assuming lambda in main.py remains at 0.5
    plt.colorbar()
    plt.show()

    cs_results = f["tallies"]["cs_tally_0"]["flux"]["mean"][:]

    mesh_results = f["tallies"]["mesh_tally_0"]["flux"]["mean"][:]
    plt.imshow(np.rot90(mesh_results), extent=[0, 60, 0, 100])
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"MC/DC Full Sim Results - $10^6$ particles")
    plt.colorbar()
    # plt.savefig('MCDC_full_sim_kobayashi.png')
    plt.show()

Nx = 30
Ny = 50
N_fine_cells = Nx * Ny

# Can use this for post-processing
mesh_b = S @ mesh_results.flatten()
# b = mesh_b

# Use this for analyzing the in-situ results
cs_b = cs_results
b = cs_b

# Constructing T and A
idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

T_inv = np.kron(idct_basis_y, idct_basis_x)
A = S @ T_inv


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
fig, axes = plt.subplots(4, 4, figsize=(10, 12))

# Populate the subplots
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        im = ax.imshow(
            np.rot90(reconstruct(l_array[i * 4 + j])), extent=[0, 60, 0, 100]
        )
        ax.set_title(f"$\lambda$ = {l_array[i * 4 + j]:.5g}")

        if j == 0:
            ax.set_ylabel("y [cm]")
        if i == 3:
            ax.set_xlabel("x [cm]")

        # Add a colorbar to the subplot
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=1)
        cbar.formatter.set_powerlimits((0, 0))

# Adjust layout
plt.suptitle(
    "Basis Pursuit Denoising Reconstructions with Different Values of $\lambda$",
    fontsize=16,
)
plt.tight_layout()
# plt.savefig('BPDN_lambda_testing_kobayashi.png')
plt.show()

rel_norms = np.ones(len(l_array))

for i in range(len(l_array)):
    rel_norms[i] = rel_norm(reconstruct(l_array[i]), mesh_results)

plt.plot(l_array, rel_norms)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("Relative L$^2$ Error")
plt.title("Relative Error vs $\lambda$ - Kobayashi Reconstructions")
plt.tight_layout()
# plt.savefig('Relative_error_vs_lambda_kobayashi.png')
plt.show()
