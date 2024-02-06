import numpy as np
import os, sys, argparse
from task import task

# Option parser
parser = argparse.ArgumentParser(description="MC/DC verification test")
parser.add_argument("--mpiexec", type=int, default=0)
parser.add_argument("--srun", type=int, default=0)
args, unargs = parser.parse_known_args()
mpiexec = args.mpiexec
srun = args.srun


# =============================================================================
# Functions
# =============================================================================


def run(N_hist, name):
    """
    N_hist: number of histories
    name: name for the job
    """
    output = "output_%i" % N_hist

    if srun > 1:
        os.system(
            "srun -n %i python input.py --mode=numba --N_particle=%i --output=%s"
            % (srun, N_hist, output)
        )
    elif mpiexec > 1:
        os.system(
            "mpiexec -n %i python input.py --mode=numba --N_particle=%i --output=%s"
            % (mpiexec, N_hist, output)
        )
    else:
        os.system(
            "python input.py --mode=numba --N_particle=%i --output=%s"
            % (N_hist, output)
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
