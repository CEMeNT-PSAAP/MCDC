import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    m1 = mcdc.material(capture=np.array([50.0]))
    m2 = mcdc.material(capture=np.array([5.0]))
    m3 = mcdc.material(capture=np.array([0.0])) # Vacuum
    m4 = mcdc.material(capture=np.array([0.1]), scatter=np.array([[0.9]]))

    s1 = mcdc.surface('plane-x', x=0.0, bc="reflective")
    s2 = mcdc.surface('plane-x', x=2.0)
    s3 = mcdc.surface('plane-x', x=3.0)
    s4 = mcdc.surface('plane-x', x=5.0)
    s5 = mcdc.surface('plane-x', x=8.0, bc="vacuum")

    mcdc.cell([+s1, -s2], m1)
    mcdc.cell([+s2, -s3], m2)
    mcdc.cell([+s3, -s4], m3)
    mcdc.cell([+s4, -s5], m4)

    mcdc.source(x=[0.0, 2.0], isotropic=True, prob=100.0)

    mcdc.source(x=[5.0, 6.0], isotropic=True, prob=1.0)

    scores = ['flux', 'flux-x']
    mcdc.tally(scores=scores, 
               x=np.linspace(0.0, 8.0, 41))

    mcdc.setting(N_particle=1E2, progress_bar=False)

    mcdc.run()
    
    # =========================================================================
    # Check output
    # =========================================================================

    output = h5py.File('output.h5', 'r')
    answer = h5py.File('answer.h5', 'r')
    for score in scores:
        name = 'tally/'+score+'/mean'
        a = output[name][:]
        b = answer[name][:]
        assert np.array_equal(a,b)
        
        name = 'tally/'+score+'/sdev'
        a = output[name][:]
        b = answer[name][:]
        assert np.array_equal(a,b)

    output.close()
    answer.close()
