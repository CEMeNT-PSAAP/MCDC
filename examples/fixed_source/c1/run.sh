#!/bin/tcsh
#SBATCH -N 8
#SBATCH -t 1:00:00


srun -n 288 python input.py
