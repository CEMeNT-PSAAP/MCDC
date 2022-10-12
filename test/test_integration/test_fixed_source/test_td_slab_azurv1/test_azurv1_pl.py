import numpy as np
import sys

import mcdc


def test_azurv1_pl():
    N_particle = 1000
    tag        = 'test_azurv1_pl' #sys.argv[3]

    # =============================================================================
    # Set model
    # =============================================================================

    # Set materials
    m = mcdc.material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                    fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

    # Set surfaces
    s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
    s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

    # Set cells
    mcdc.cell([+s1, -s2], m)

    # =============================================================================
    # Set source
    # =============================================================================

    mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

    # =============================================================================
    # Set tally, setting, and run mcdc
    # =============================================================================

    # Tally
    mcdc.tally(scores=['flux', 'flux-x', 'flux-t'], 
            x=np.linspace(-20.5, 20.5, 202), 
            t=np.linspace(0.0, 20.0, 21))

    # Setting
    mcdc.setting(N_particle=N_particle, output='output_'+tag+'_'+str(N_particle), 
                progress_bar=False)

    # Run
    mcdc.run()
