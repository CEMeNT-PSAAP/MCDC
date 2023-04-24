import numpy as np
import argparse

import mcdc

parser = argparse.ArgumentParser(description="Setting changeable parameters")
parser.add_argument(
    "--particles", action="store", dest="N_particle", type=int, default=100
)
parser.add_argument("--file", action="store", dest="output_file", default="output")
parser.add_argument("--mode", action="store", dest="mode", type=bool, default=False)
args = parser.parse_args()

# =========================================================================
# Set model and run
# =========================================================================

m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

mcdc.cell([+s1, -s2], m)

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

scores = ["flux", "flux-x", "flux-t"]
mcdc.tally(scores=scores, x=np.linspace(-20.5, 20.5, 202), t=np.linspace(0.0, 20.0, 21))

mcdc.setting(N_particle=args.N_particle, progress_bar=False, output=args.output_file)

mcdc.run()
