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
# A shielding problem based on Problem 2 of [Coper NSE 2001]
# https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

SigmaT = 5.0
c = 0.8
m_barrier = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))
SigmaT = 1.0
m_room = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))

sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=2.0)
sx3 = mcdc.surface("plane-x", x=2.4)
sx4 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=2.0)
sy3 = mcdc.surface("plane-y", y=4.0, bc="vacuum")

mcdc.cell([+sx1, -sx2, +sy1, -sy2], m_room)
mcdc.cell([+sx1, -sx4, +sy2, -sy3], m_room)
mcdc.cell([+sx3, -sx4, +sy1, -sy2], m_room)
mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)

mcdc.source(x=[0.0, 1.0], y=[0.0, 1.0], isotropic=True)

scores = ["flux", "flux-x", "flux-y"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 4.0, 40), y=np.linspace(0.0, 4.0, 40))

mcdc.setting(N_particle=args.N_particle, progress_bar=False, output=args.output_file)
mcdc.implicit_capture()

mcdc.run()
