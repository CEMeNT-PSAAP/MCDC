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
    y = f["tally/grid/y"][:]
    dy = y[1] - y[0]
    y_mid = 0.5 * (y[:-1] + y[1:])

# =============================================================================
# Plot
# =============================================================================

# Load output
with h5py.File("output.h5", "r") as f:
    # Note the spatial (dx) and source strength (100+1) normalization
    phi = f["tally/flux/mean"][:] / dx *dy
    phi_sd = f["tally/flux/sdev"][:] / dx *dy
    J = f["tally/current/mean"][:,0] / dx *dy
    J_sd = f["tally/current/sdev"][:,0] / dx *dy
    plt.pcolormesh(phi)
    plt.show()
    #print("Keys: %s" % f["tally/flux/mea"].keys())
    domain_mesh_x = f["input_deck/technique/domain_mesh"]["z"][:]
    domain_mesh_y = f["input_deck/technique/domain_mesh"]["y"][:]
    domain_mesh_z = f["input_deck/technique/domain_mesh"]["z"][:]
    work_ratio = f["input_deck/technique/work_ratio"][:]
    max_buff = f["input_deck/technique/exchange_rate"][()]
    N_part = f["input_deck/setting/N_particle"][()]
    N_dom =len(work_ratio)
    N_proc = sum(work_ratio)


file = "dmp_2d.csv"
f = open(file,'w')
# Writing run params:

f.write("Domain Decomp info:\n Number of domains:,"+str(N_dom)+"\n Domain mesh:\n")
for i in range(len(domain_mesh_x)):
    line=str(domain_mesh_x[i])+','
    f.write(line)
f.write('\n')
f.write("Number of processors:,"+str(N_proc)+"\nProcessors per domain:\n")
for i in range(len(work_ratio)):
    line=str(work_ratio[i])+','
    f.write(line)
line = '\n Maximum buffer size:,'+str(max_buff)+'\n'
f.write(line)
line = '\n Number of particles run:,'+str(N_part)+'\n'
f.write(line)
f.write("\n x,phi,phisd,J,Jsd \n")
for i in range(len(phi)):
    line = str(x[i])+','
    f.write(line)
    for j in range(len(phi[0])):
        line =  str(phi[i,j])+','
        f.write(line)
    line = '\n'
    f.write(line)
f.close()
