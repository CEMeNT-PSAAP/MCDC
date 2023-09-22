import numpy as np

import mcdc  
m = mcdc.material(capture=np.array([0.0]))

s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=3.0, bc="vacuum")
s3 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s4 = mcdc.surface("plane-z", z=3.0, bc="vacuum")

mcdc.cell([+s1,+s3,-s4, -s2], m)
#mcdc.source(x=[0.0, 3.0],z=[0.0,1e-10],direction=[0.0, 0.0, 1.0])
#mcdc.source(x=[0.0, 1e-10],z=[0.0,3.0],direction=[1.0, 0.0, 0.0])
#mcdc.source(x=[0.0, 1.0],z=[0.0,0.01],direction=[0.0, 0.0, 1.0])#  isotropic=True,prob=1)#direction=[1.0, 0.0, 0.0])
mcdc.source(z=[0.0, 1.0],x=[0.0,2.0], isotropic=True,prob=1)#direction=[1.0, 0.0, 0.0])
#mcdc.source(point=[1e-10, 0.0, 1e-10], direction=[1.0, 0.0, 0.0])
#mcdc.source(point=[1e-10, 0.0, 1.0], direction=[1.0, 0.0, 0.0])
#mcdc.source(point=[1e-10, 0.0, 2.0], direction=[1.0, 0.0, 0.0])
#mcdc.source(point=[1e-10, 0.0, 1e-10], direction=[0.0, 0.0, 1.0])
#mcdc.source(point=[1.0, 0.0, 1e-10], direction=[0.0, 0.0, 1.0])
#mcdc.source(point=[2.0, 0.0, 1e-10], direction=[0.0, 0.0, 1.0])




scores = ["flux","current"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 3.0, 31), z = np.linspace(0.0, 3.0, 31))

mcdc.setting(N_particle=1e4)
mcdc.domain_decomp(z=np.linspace(0.0,3.0,4),x=np.linspace(0.0,3.0,4),work_ratio=([2,1,1,1,1,1,1,1,1]))#,x=np.linspace(0.0,3.0,4)
#,y=np.linspace(0.0,4.0,3),x=np.linspace(0.0,4.0,3),,z=np.linspace(0.0,4.0,3)
mcdc.run()
