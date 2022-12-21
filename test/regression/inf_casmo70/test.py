import numpy as np
import h5py

import mcdc

def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    with np.load('CASMO-70.npz') as data:
        SigmaC = data['SigmaC']*1.5 # /cm
        SigmaS = data['SigmaS']
        SigmaF = data['SigmaF']
        nu_p   = data['nu_p']
        nu_d   = data['nu_d']
        chi_p  = data['chi_p']
        chi_d  = data['chi_d']
        G      = data['G']
        speed    = data['v']
        lamd     = data['lamd']

    m = mcdc.material(capture=SigmaC, scatter=SigmaS, fission=SigmaF, nu_p=nu_p,
                      chi_p=chi_p, nu_d=nu_d, chi_d=chi_d)

    s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
    s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

    c = mcdc.cell([+s1, -s2], m)

    energy = np.zeros(G); energy[-1] = 1.0
    source = mcdc.source(energy=energy)

    scores = ['flux']
    mcdc.tally(scores=scores)

    mcdc.setting(N_particle=1E1, progress_bar=False)

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
