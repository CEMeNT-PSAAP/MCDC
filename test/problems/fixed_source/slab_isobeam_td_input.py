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

m = mcdc.material(capture=np.array([1.0]))

s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=5.0, bc="vacuum")

mcdc.cell([+s1, -s2], m)

mcdc.source(point=[1e-10, 0.0, 0.0], time=[0.0, 5.0], white_direction=[1.0, 0.0, 0.0])

scores = ["flux", "flux-x", "flux-t"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 5.0, 51), t=np.linspace(0.0, 5.0, 51))

mcdc.setting(N_particle=args.N_particle, progress_bar=False, output=args.output_file)

mcdc.run()
