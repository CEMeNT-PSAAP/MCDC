#!/bin/tcsh
#SBATCH -N 16
#SBATCH -t 8:00:00

srun -n 576 python input.py
