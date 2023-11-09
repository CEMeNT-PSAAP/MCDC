import numpy as np

import mcdc

m = mcdc.material(capture=np.array([0.0]))

s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=5.0, bc="vacuum")

mcdc.cell([+s1, -s2], m)

mcdc.source(point=[1e-10, 0.0, 0.0], direction=[1.0, 0.0, 0.0])

scores = ["flux", "current"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 5.0, 51))

mcdc.setting(N_particle=10, active_bank_buff=1000000)
mcdc.domain_decomp(x=np.linspace(0.0, 5.0, 6))

mcdc.run()
