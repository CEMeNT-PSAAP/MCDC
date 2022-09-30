#!/bin/tcsh
#SBATCH -N 8
#SBATCH -p pdebug
#SBATCH -t 1:00:00

srun -n 288 python input.py
