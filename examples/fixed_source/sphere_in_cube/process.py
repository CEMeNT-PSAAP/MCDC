import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import cvxpy as cp

# Load results
with h5py.File("output.h5", "r") as f:
    # trying to compare cs results and mesh results
    center_points = np.load("center_points.npy")
    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]
    cs_alphas = cs_results[:-1] / np.max(cs_results[:-1])
    # plt.scatter(center_points[0][:-1], center_points[1][:-1], alpha=cs_alphas)
    # plt.show()

    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    plt.imshow(cs_results[:-1].reshape(mesh_results.shape))
    plt.title("cs")
    plt.show()
    plt.imshow(mesh_results)
    plt.title("mesh")
    plt.show()

    print(f"cs_results = {cs_results}")
    print(f"sh_results = {mesh_results.flatten()}")


Nx = 40
Ny = 40
S = np.load("sphere_S.npy")
mesh_b = S @ mesh_results.flatten()
cs_b = cs_results


# b = mesh_b

# print(f'shape of mesh b = {mesh_b.shape}')
# print(f'shape of cs b = {cs_b.shape}')

# idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
# idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

# T_inv = np.kron(idct_basis_y, idct_basis_x)
# A = S @ T_inv
# N_fine_cells = 1600

# vx = cp.Variable(N_fine_cells)
# l = 0
# objective = cp.Minimize(0.5 * cp.norm(A @ vx - b, 2) + l * cp.norm(vx, 1))
# prob = cp.Problem(objective)
# result = prob.solve(verbose=False)
# sparse_solution = np.array(vx.value).squeeze()

# recon = T_inv @ sparse_solution
# recon_reshaped = recon.reshape(Ny, Nx)

# plt.imshow(recon_reshaped)
# plt.title('reconstruction')
# plt.show()
