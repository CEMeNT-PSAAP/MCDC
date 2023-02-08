import numpy as np
import h5py, sys

import mcdc


def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    lib = h5py.File('c5g7.h5', 'r')
    def set_mat(mat):
        return mcdc.material(capture = mat['capture'][:],
                             scatter = mat['scatter'][:],
                             fission = mat['fission'][:],
                             nu_p    = mat['nu_p'][:],
                             nu_d    = mat['nu_d'][:],
                             chi_p   = mat['chi_p'][:],
                             chi_d   = mat['chi_d'][:],
                             speed   = mat['speed'],
                             decay   = mat['decay'],
                             sensitivity=True)


    mat_uo2 = set_mat(lib['uo2']) # Fuel: UO2
    mat_mod = set_mat(lib['mod']) # Moderator
    mat_cr  = set_mat(lib['cr']) # Control rod

    s1 = mcdc.surface('plane-x', x=0.0, bc="reflective")
    s2 = mcdc.surface('plane-x', x=0.5, sensitivity=True)
    s3 = mcdc.surface('plane-x', x=1.5, sensitivity=True)
    s4 = mcdc.surface('plane-x', x=2.0, bc="reflective")

    mcdc.cell([+s1, -s2], mat_uo2)
    mcdc.cell([+s2, -s3], mat_mod)
    mcdc.cell([+s3, -s4], mat_cr)

    mcdc.source(point=[1.0, 0.0, 0.0], energy=[1,0,0,0,0,0,0], isotropic=True)

    scores = ['flux']
    mcdc.tally(scores=['flux'], x=np.linspace(0.0, 2.0, 11))

    mcdc.setting(N_particle=5)

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
        assert np.isclose(a,b).all()
        
        name = 'tally/'+score+'/sdev'
        a = output[name][:]
        b = answer[name][:]
        assert np.isclose(a,b).all()

    output.close()
    answer.close()
