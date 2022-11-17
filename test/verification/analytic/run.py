import numpy as np
import os
import sys

if len(sys.argv) > 1:
    N_proc = int(sys.argv[1])
else:
    N_proc = 1

# Fixed source
N_min = 3
N_max = 7
for task in os.scandir('./fixed_source'):
    os.chdir(task)
    for N_hist in np.logspace(N_min, N_max, (N_max-N_min)*2+1):
        if not os.path.isfile('output_'+str(int(N_hist))+'.h5'):
            print(task, int(N_hist))
            if N_proc == 1:
                os.system("python input.py --mode=numba %i"%(N_hist))
            else:
                os.system("srun -n %i python input.py --mode=numba %i"%(N_proc,N_hist))
    os.chdir(r"../..")

# Eigenvalue
'''
N_min = 1
N_max = 3
for task in os.scandir('./eigenvalue'):
    os.chdir(task)
    for N_hist in np.logspace(N_min, N_max, (N_max-N_min)*4+1):
        print(task, int(N_hist))
        if N_proc == 1:
            os.system("python input.py --mode=numba %i"%(N_hist))
        else:
            os.system("srun -n %i python input.py --mode=numba %i"%(N_proc,N_hist))
    os.chdir(r"../..")
'''
