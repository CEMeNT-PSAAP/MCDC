import numpy as np
import argparse

import mcdc

parser = argparse.ArgumentParser(description="Setting changeable parameters")
parser.add_argument(
    "--particles", action="store", dest="N_particle", type=int, default=100
)
parser.add_argument(
    "--file", action="store", dest="output_file", type=str, default="output"
)
parser.add_argument("--mode", action="store", dest="mode", type=bool, default=False)
args = parser.parse_args()

# =========================================================================
# Set model and run
# =========================================================================

m1 = mcdc.material(capture=np.array([1.0]))
m2 = mcdc.material(capture=np.array([1.5]))
m3 = mcdc.material(capture=np.array([2.0]))

s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s2 = mcdc.surface("plane-z", z=2.0)
s3 = mcdc.surface("plane-z", z=4.0)
s4 = mcdc.surface("plane-z", z=6.0, bc="vacuum")

mcdc.cell([+s1, -s2], m2)
mcdc.cell([+s2, -s3], m3)
mcdc.cell([+s3, -s4], m1)

mcdc.source(z=[0.0, 6.0], isotropic=True)

scores = ["flux", "current", "flux-z", "current-z"]
mcdc.tally(
    scores=scores, z=np.linspace(0.0, 6.0, 61), mu=np.linspace(-1.0, 1.0, 32 + 1)
)

mcdc.setting(N_particle=args.N_particle, progress_bar=False, output=args.output_file)

mcdc.run()
