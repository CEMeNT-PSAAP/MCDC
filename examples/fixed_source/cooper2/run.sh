#!/bin/tcsh
#SBATCH -N 32
#SBATCH -t 8:00:00

srun -n 1152 python input.py --mode=numba
