import numpy as np
import h5py
import os

for N_hist in np.logspace(3, 10, 15):
    os.system("srun -n 180 python input.py --mode=numba %i"%N_hist)
