import numpy as np
import os, sys
from task import task


if len(sys.argv) > 1:
    N_rank = int(sys.argv[1])
else:
    N_rank = 1


# =============================================================================
# Functions
# =============================================================================


def run(N_hist, name):
    """
    N_hist: number of histories
    name: name for the job
    """
    output = "output_%i" % N_hist

    os.system(
        "srun -n %i python input.py --mode=numba --N_particle=%i --output=%s"
        % (N_rank, N_hist, output)
    )


# =============================================================================
# Select and run tests
# =============================================================================

for name in task.keys():
    os.chdir(name)
    N_min = task[name]["N_lim"][0]
    N_max = task[name]["N_lim"][1]
    N = task[name]["N"]
    for N_hist in np.logspace(N_min, N_max, N):
        N_hist = int(N_hist)
        print(name, N_hist)
        run(N_hist, name)
    os.chdir(r"..")
