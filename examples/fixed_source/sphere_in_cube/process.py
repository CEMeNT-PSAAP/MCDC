import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import cvxpy as cp

# Note: there are some lines in main.py with np.save(...) that I added
# for ease of post-processing, like getting the center points used and
# the sampling matrix S. None are required for the input file and this
# script to run, but may be useful for debugging purposes

with h5py.File("output.h5", "r") as f:
    S = f["tallies"]["cs_tally_0"]["S"][:]
    recon = f["tallies"]["cs_tally_0"]["fission"]["reconstruction"]
    plt.imshow(recon)
    plt.title("Reconstruction, $\lambda$ = 0.5")  # assuming l in main.py remains at 0.5
    plt.colorbar()
    plt.show()

    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]

    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    plt.imshow(mesh_results)
    plt.title("mesh results")
    plt.colorbar()
    plt.show()

Nx = 40
Ny = 40
N_fine_cells = Nx * Ny

# Can use this for post-processing
# mesh_b = S @ mesh_results.flatten()
# b = mesh_b

# Use this for analyzing the in-situ results
cs_b = cs_results
b = cs_b

# Constructing T and A
idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

T_inv = np.kron(idct_basis_y, idct_basis_x)
A = S @ T_inv

# Basis pursuit denoising solver - change l to get different results
vx = cp.Variable(N_fine_cells)
l = 10
objective = cp.Minimize(0.5 * cp.norm(A @ vx - b, 2) + l * cp.norm(vx, 1))
prob = cp.Problem(objective)
result = prob.solve(verbose=False)
sparse_solution = np.array(vx.value).squeeze()

# Obtaining the reconstruction
recon = T_inv @ sparse_solution
recon_reshaped = recon.reshape(Ny, Nx)

plt.imshow(recon_reshaped)
plt.title(f"Reconstruction, $\lambda$ = {l}")
plt.colorbar()
plt.show()
