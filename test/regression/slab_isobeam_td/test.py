import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    m = mcdc.material(capture=np.array([1.0]))

    s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
    s2 = mcdc.surface('plane-x', x=5.0, bc="vacuum")

    mcdc.cell([+s1, -s2], m)

    mcdc.source(point=[1E-10,0.0,0.0], time=[0.0, 5.0], 
                white_direction=[1.0, 0.0, 0.0])

    scores = ['flux', 'flux-x', 'flux-t']
    mcdc.tally(scores=scores, x=np.linspace(0.0, 5.0, 51), 
               t=np.linspace(0.0, 5.0, 51))

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
