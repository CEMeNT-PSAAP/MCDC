import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad

# =============================================================================
# Reference solution (not accurate enough for N_hist > 1E7)
# =============================================================================


with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])

# =============================================================================
# Plot
# =============================================================================

# Load output
with h5py.File("output.h5", "r") as f:
    # Note the spatial (dx) and source strength (100+1) normalization
    phi = f["tally/flux/mean"][:] / dx
    phi_sd = f["tally/flux/sdev"][:] / dx
    J = f["tally/current/mean"][:, 0] / dx
    J_sd = f["tally/current/sdev"][:, 0] / dx
    # print("Keys: %s" % f["input_deck/technique"].keys())
    domain_mesh_x = f["input_deck/technique/domain_mesh"]["z"][:]
    domain_mesh_y = f["input_deck/technique/domain_mesh"]["y"][:]
    domain_mesh_z = f["input_deck/technique/domain_mesh"]["z"][:]
    work_ratio = f["input_deck/technique/work_ratio"][:]
    max_buff = f["input_deck/technique/exchange_rate"][()]
    N_part = f["input_deck/setting/N_particle"][()]
    N_dom = len(work_ratio)
    N_proc = sum(work_ratio)


file = "domain_test_results.csv"
f = open(file, "w")
# Writing run params:

f.write("Domain Decomp info:\n Number of domains:," + str(N_dom) + "\n Domain mesh:\n")
for i in range(len(domain_mesh_x)):
    line = str(domain_mesh_x[i]) + ","
    f.write(line)
f.write("\n")
f.write("Number of processors:," + str(N_proc) + "\nProcessors per domain:\n")
for i in range(len(work_ratio)):
    line = str(work_ratio[i]) + ","
    f.write(line)
line = "\n Maximum buffer size:," + str(max_buff) + "\n"
f.write(line)
line = "\n Number of particles run:," + str(N_part) + "\n"
f.write(line)
f.write("\n x,phi,phisd,J,Jsd \n")
for i in range(len(phi)):
    line = (
        str(x[i])
        + ","
        + str(phi[i])
        + ","
        + str(phi_sd[i])
        + ","
        + str(J[i])
        + ","
        + str(J_sd[i])
        + "\n"
    )
    f.write(line)
f.close()
# Flux - spatial average
plt.plot(x_mid, phi, "-b", label="MC")
plt.fill_between(x_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
plt.xlabel(r"$x$, cm")
plt.ylabel("Flux")
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()
