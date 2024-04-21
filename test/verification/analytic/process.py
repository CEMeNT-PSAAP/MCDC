import numpy as np
import os
from task import task

<<<<<<< HEAD
# Fixed source
N_min = 3
N_max = 5
for task in os.scandir("./fixed_source"):
    print(task)
    os.chdir(task)
    os.system("python process.py %i %i" % (N_min, N_max))
    os.chdir(r"../..")
=======
>>>>>>> main

for name in task.keys():
    os.chdir(name)
    N_min = task[name]["N_lim"][0]
    N_max = task[name]["N_lim"][1]
    N = task[name]["N"]
    os.system("python process.py %i %i %i" % (N_min, N_max, N))
    os.chdir(r"..")
