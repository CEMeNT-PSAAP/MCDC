import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    m      = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]))
    m_void = mcdc.material(capture=np.array([5E-5]), scatter=np.array([[5E-5]]))

    sx1 = mcdc.surface('plane-x', x=0.0,  bc="reflective")
    sx2 = mcdc.surface('plane-x', x=10.0)
    sx3 = mcdc.surface('plane-x', x=30.0)
    sx4 = mcdc.surface('plane-x', x=40.0)
    sx5 = mcdc.surface('plane-x', x=60.0, bc="vacuum")
    sy1 = mcdc.surface('plane-y', y=0.0,  bc="reflective")
    sy2 = mcdc.surface('plane-y', y=10.0)
    sy3 = mcdc.surface('plane-y', y=50.0)
    sy4 = mcdc.surface('plane-y', y=60.0)
    sy5 = mcdc.surface('plane-y', y=100.0, bc="vacuum")
    sz1 = mcdc.surface('plane-z', z=0.0,  bc="reflective")
    sz2 = mcdc.surface('plane-z', z=10.0)
    sz3 = mcdc.surface('plane-z', z=30.0)
    sz4 = mcdc.surface('plane-z', z=40.0)
    sz5 = mcdc.surface('plane-z', z=60.0, bc="vacuum")

    mcdc.cell([+sx1, -sx2, +sy1, -sy2, +sz1, -sz2], m)
    mcdc.cell([+sx1, -sx2, +sy2, -sy3, +sz1, -sz2], m_void)
    mcdc.cell([+sx1, -sx3, +sy3, -sy4, +sz1, -sz2], m_void)
    mcdc.cell([+sx3, -sx4, +sy3, -sy4, +sz1, -sz3], m_void)
    mcdc.cell([+sx3, -sx4, +sy3, -sy5, +sz3, -sz4], m_void)
    mcdc.cell([+sx1, -sx3, +sy1, -sy5, +sz2, -sz5], m)
    mcdc.cell([+sx2, -sx5, +sy1, -sy3, +sz1, -sz2], m)
    mcdc.cell([+sx3, -sx5, +sy1, -sy3, +sz2, -sz5], m)
    mcdc.cell([+sx3, -sx5, +sy4, -sy5, +sz1, -sz3], m)
    mcdc.cell([+sx4, -sx5, +sy4, -sy5, +sz3, -sz5], m)
    mcdc.cell([+sx4, -sx5, +sy3, -sy4, +sz1, -sz5], m)
    mcdc.cell([+sx3, -sx4, +sy3, -sy5, +sz4, -sz5], m)
    mcdc.cell([+sx1, -sx3, +sy4, -sy5, +sz1, -sz2], m)

    mcdc.source(x=[0.0, 10.0], y=[0.0, 10.0], z=[0.0, 10.0], time=[0.0, 50.0],
                isotropic=True)

    scores = ['flux']
    mcdc.tally(scores=scores,
               x=np.linspace(0.0, 60.0, 61), y=np.linspace(0.0, 100.0, 101),
               t=np.linspace(0.0, 200.0, 21))

    mcdc.setting(N_particle=1E2)
    mcdc.implicit_capture()

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
    assert True
