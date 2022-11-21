import numpy as np
import h5py

import mcdc

def test():
    # =============================================================================
    # Set model and run
    # =============================================================================

    m     = mcdc.material(capture=np.array([0.1]), scatter=np.array([[0.9]]))
    m_abs = mcdc.material(capture=np.array([0.9]), scatter=np.array([[0.1]]))

    s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
    s2 = mcdc.surface('plane-x', x=np.array([2.0, 2.0,  5.0,  1.0]),
                                 t=np.array([0.0, 5.0, 10.0, 10.0]))
    s3 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

    mcdc.cell([+s1, -s2], m)
    mcdc.cell([+s2, -s3], m_abs)

    mcdc.source(x=[0.0, 6.0], time=[0.0, 10.0], isotropic=True)

    scores = ['flux']
    mcdc.tally(scores=scores,
               x=np.linspace(0.0, 6.0, 61), t=np.linspace(0.0, 15.0, 151))

    mcdc.setting(N_particle=int(1E2))

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
