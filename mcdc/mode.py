import argparse
from numba import config


# Parse arguments
parser = argparse.ArgumentParser(description='MC/DC: Monte Carlo Dynamic Code')
parser.add_argument('--mode', type=str, help='run mode',
                    choices=['python', 'numba'], default='python')
args, unargs = parser.parse_known_args()

# Set mode
mode = args.mode
if mode == 'python':
    config.DISABLE_JIT = True
elif mode == 'numba':
    config.DISABLE_JIT = False
