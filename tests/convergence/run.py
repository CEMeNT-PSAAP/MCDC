import numpy as np
import os
import sys

cases  = ['slab_parallel_beam', 'slab_uniform_source', 'slab_reed', 
          'inf_SHEM361', 'td_slab_azurv1']
caseks = ['k_slab_kornreich', 'k_inf_SHEM361']
N_proc = int(sys.argv[1])
N_max  = 7

for case in cases:
    os.chdir(r"./"+case)
    for N_hist in np.logspace(3, N_max, (N_max-3)*2+1):
        os.system("srun -n %i python input.py --mode=numba %i"%(N_proc,N_hist))
    os.system("python process.py %i"%N_max)
    os.chdir(r"..")

for case in caseks:
    os.chdir(r"./"+case)
    os.system("srun -n %i python input.py --mode=numba"%(N_proc))
    os.system("python process.py")
    os.chdir(r"..")
