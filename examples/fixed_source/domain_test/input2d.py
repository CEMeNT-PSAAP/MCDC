import numpy as np

import mcdc  
m = mcdc.material(capture=np.array([0.5]))

s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
s3 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s4 = mcdc.surface("plane-z", z=4.0, bc="vacuum")


mcdc.cell([+s1,+s3,-s4, -s2], m)

mcdc.source(x=[0.0, 3.0],z=[0.0,2.0], isotropic=True)

scores = ["flux","current"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 4.0, 21), z = np.linspace(0.0, 4.0, 21))

mcdc.setting(N_particle=1e4,active_bank_buff=1000000)
mcdc.domain_decomp(z=np.linspace(0.0,4.0,3),exchange_rate=10,work_ratio=([1,1]))
#,y=np.linspace(0.0,4.0,3),x=np.linspace(0.0,4.0,3),,z=np.linspace(0.0,4.0,3)
mcdc.run()
