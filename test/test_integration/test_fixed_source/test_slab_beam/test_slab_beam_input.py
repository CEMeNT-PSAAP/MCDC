import numpy as np
import sys

import mcdc

def test_slab_beam():
    N_particle = 1000
    tag        = 'test_slab_beam'

    # =============================================================================
    # Set model
    # =============================================================================
    # Three slab layers with different materials

    # Set materials
    m1 = mcdc.material(capture=np.array([1.0]))
    m2 = mcdc.material(capture=np.array([1.5]))
    m3 = mcdc.material(capture=np.array([2.0]))

    # Set surfaces
    s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
    s2 = mcdc.surface('plane-x', x=2.0)
    s3 = mcdc.surface('plane-x', x=4.0)
    s4 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

    # Set cells
    mcdc.cell([+s1, -s2], m2)
    mcdc.cell([+s2, -s3], m3)
    mcdc.cell([+s3, -s4], m1)

    # =============================================================================
    # Set source
    # =============================================================================
    # Parallel beam, at x=0, going in the +x direction
    # Source needs to be slightly shifted to the right to make sure it starts in
    # the right cell.

    mcdc.source(point=[1E-10,0.0,0.0], direction=[1.0,0.0,0.0])

    # =============================================================================
    # Set tally, setting, and run mcdc
    # =============================================================================

    # Tally
    mcdc.tally(scores=['flux', 'flux-x'], x=np.linspace(0.0, 6.0, 61))

    # Setting
    mcdc.setting(N_particle=N_particle, output='output_'+tag+'_'+str(N_particle), 
                progress_bar=False)

    # Run
    mcdc.run()
