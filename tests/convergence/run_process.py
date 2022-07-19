import numpy as np
import os
import sys

# Fixed source
N_min = 3
N_max = 4#7
for task in os.scandir('./fixed_source'):
    print(task)
    os.chdir(task)
    os.system("python process_convergence.py %i %i"%(N_min,N_max))
    os.chdir(r"../..")

# Eigenvalue
N_min = 1
N_max = 2#3
for task in os.scandir('./eigenvalue'):
    print(task)
    os.chdir(task)
    os.system("python process_convergence.py %i %i"%(N_min,N_max))
    os.chdir(r"../..")
