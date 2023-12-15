import numpy as np
import h5py

import mcdc


# =============================================================================
# Set model and run
# =============================================================================

m1 = mcdc.material(capture=np.array([0.90]))
m2 = mcdc.material(capture=np.array([0.15]))
m3 = mcdc.material(capture=np.array([0.60]))

s1 = mcdc.surface("plane-x", x=-1.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=2.0)
s3 = mcdc.surface("plane-x", x=5.0)
s4 = mcdc.surface("plane-x", x=6.0, bc="vacuum")

mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)
mcdc.cell([+s3, -s4], m3)

mcdc.source(point=[0.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0])

scores = ["exit"]
mcdc.tally(scores=scores, x=np.linspace(0.0, 6.0, 2))

mcdc.setting(N_particle=1e1, N_batch=1e1, progress_bar=False)

mcdc.uq(material=m1, distribution="uniform", capture=np.array([0.7]))
mcdc.uq(material=m2, distribution="uniform", capture=np.array([0.12]))
mcdc.uq(material=m3, distribution="uniform", capture=np.array([0.5]))

mcdc.run()

# =========================================================================
# Check output
# =========================================================================

output = h5py.File("output.h5", "r")
answer = h5py.File("answer.h5", "r")
for score in scores:
    name = "tally/" + score + "/mean"
    a = output[name][:]
    b = answer[name][:]
    assert np.isclose(a, b).all()

    name = "tally/" + score + "/sdev"
    a = output[name][:]
    b = answer[name][:]
    assert np.isclose(a, b).all()

    name = "tally/" + score + "/uq_var"
    a = output[name][:]
    b = answer[name][:]
    assert np.isclose(a, b).all()

output.close()
answer.close()
