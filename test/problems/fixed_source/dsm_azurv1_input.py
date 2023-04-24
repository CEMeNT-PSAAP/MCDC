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

n1 = mcdc.nuclide(capture=np.array([0.5]))
n2 = mcdc.nuclide(
    capture=np.array([0.1]),
    fission=np.array([0.4]),
    nu_p=np.array([2.5]),
    sensitivity=True,
)

m = mcdc.material(nuclides=[(n1, 1.0), (n2, 1.0)])

s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

mcdc.cell([+s1, -s2], m)

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

scores = ["flux-t"]
mcdc.tally(
    scores=scores,
    x=np.linspace(-20.0, 20.0, 202),
    t=np.linspace(0.0, 20.0, 21),
)

mcdc.setting(N_particle=args.N_particle, progress_bar=False, output=args.output_file)


mcdc.run()
