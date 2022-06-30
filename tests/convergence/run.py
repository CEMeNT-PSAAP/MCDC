import numpy as np
import os
import sys

# Parameters
N_proc = int(sys.argv[1])
N_max  = 7

# Fixed source
for task in os.scandir('./fixed_source'):
    print(task)
    os.chdir(task)
    for N_hist in np.logspace(3, N_max, (N_max-3)*2+1):
        os.system("srun -n %i python input.py --mode=numba %i"%(N_proc,N_hist))
    os.system("python process.py %i"%N_max)
    os.chdir(r"../..")

# Eigenvalue
for task in os.scandir('./eigenvalue'):
    print(task)
    os.chdir(task)
    os.system("srun -n %i python input.py --mode=numba"%(N_proc))
    os.system("python process.py")
    os.chdir(r"../..")
