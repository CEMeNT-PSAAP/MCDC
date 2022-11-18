import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    m1 = mcdc.material(capture=np.array([1.0]))
    m2 = mcdc.material(capture=np.array([1.5]))
    m3 = mcdc.material(capture=np.array([2.0]))

    s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
    s2 = mcdc.surface('plane-x', x=2.0)
    s3 = mcdc.surface('plane-x', x=4.0)
    s4 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

    mcdc.cell([+s1, -s2], m2)
    mcdc.cell([+s2, -s3], m3)
    mcdc.cell([+s3, -s4], m1)

    mcdc.source(x=[0.0, 6.0], isotropic=True)

    scores = ['flux', 'current', 'flux-z', 'current-z']
    mcdc.tally(scores=scores, 
               z=np.linspace(0.0, 6.0, 61),
               mu=np.linspace(-1.0, 1.0, 32+1))

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
