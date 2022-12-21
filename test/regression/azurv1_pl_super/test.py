import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    m = mcdc.material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                    fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

    s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
    s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

    mcdc.cell([+s1, -s2], m)

    mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

    scores = ['flux', 'flux-x', 'flux-t']
    mcdc.tally(scores=scores, 
            x=np.linspace(-20.5, 20.5, 202), 
            t=np.linspace(0.0, 20.0, 21))

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
