# Argument parser
import argparse
parser = argparse.ArgumentParser(description='MC/DC: Monte Carlo Dynamic Code')
parser.add_argument('--mode', type=str, help='run mode', 
                    choices=['python', 'numba'], default='python')
args, unargs = parser.parse_known_args()

# Set mode
from numba import config
mode = args.mode
if mode == 'python':
    config.DISABLE_JIT = True
elif mode == 'numba':
    config.DISABLE_JIT = False

# User interface
from mcdc.input_ import *
from mcdc.main   import run
