import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load results
with h5py.File("output.h5", "r") as f:
    # trying to compare cs results and mesh results
    center_points = np.load("center_points.npy")
    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]
    cs_alphas = cs_results[:-1] / np.max(cs_results[:-1])

    # plt.scatter(center_points[0][:-1], center_points[1][:-1], alpha=cs_alphas)
    # plt.show()

    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    # plt.imshow(mesh_results)
    # plt.show()
