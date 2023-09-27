import numpy as np

import mcdc

m1 = mcdc.material(capture=np.array([0.0]))
m2 = mcdc.material(capture=np.array([3.5]) )

s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=3.0, bc="vacuum")

s3 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s4 = mcdc.surface("plane-z", z=3.0, bc="vacuum")



mcdc.cell([+s1,-s2,+s3,-s4],m1)
mcdc.source(x=[0.0, 3.0],z=[0.0,1e-10],isotropic=True, prob=1)  
mcdc.source(x=[0.0, 1e-10],z=[0.0,3.0],isotropic=True, prob=1) 


scores = ["flux", "current"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 3.0, 4), z=np.linspace(0.0, 3.0, 4))

mcdc.setting(N_particle=1e5)
mcdc.domain_decomp(z=np.linspace(0.0, 3.0, 4),x=np.linspace(0.0,3.0,4)) 
mcdc.run()
