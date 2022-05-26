import numba as nb
import numpy as np

import mcdc.kernel as kernel
import mcdc.mpi    as mpi

from mcdc.constant import *
from mcdc.print_   import print_progress, print_bank

# =========================================================================
# Source loop
# =========================================================================

@nb.njit
def loop_source(mcdc):
    # Rebase rng skip_ahead seed
    mcdc.rng.skip_ahead_strides(mpi.work_start)
    mcdc.rng.rebase()

    # Loop over particle sources
    for work_idx in range(mpi.work_size):
        # Initialize RNG wrt work index
        mcdc.rng.skip_ahead_strides(work_idx)

        # Get a source particle and put into history bank
        if mcdc.bank_source['size'] == 0:
            # Sample source
            if len(mcdc.sources) > 1:
                xi  = mcdc.rng.random()
                tot = 0.0
                for S in mcdc.sources:
                    tot += S.prob
                    if tot >= xi:
                        break
            else:
                S = mcdc.sources[0]
            P = S.get_particle(mcdc)
            kernel.set_cell(P, mcdc)
        else:
            P = mcdc.bank_source['particles'][work_idx]
        kernel.add_particle(P, mcdc.bank_history)

        # Run the source particle and its secondaries
        # (until history bank is exhausted)
        while mcdc.bank_history['size'] > 0:
            # Get particle from history bank
            P = kernel.pop_particle(mcdc.bank_history)

            # Apply weight window
            if mcdc.setting.weight_window:
                mcdc.weight_window.apply_(P, mcdc)
            
            # Particle loop
            loop_particle(P, mcdc)

        # Tally history closeout
        mcdc.tally.closeout_history()
        
        # Progress printout
        if mcdc.setting.progress_bar:
            with nb.objmode():
                print_progress(work_idx)

        
# =========================================================================
# Particle loop
# =========================================================================

@nb.njit
def loop_particle(P, mcdc):
    while P['alive']:
        # Determine and move to event
        event = kernel.move_to_event(P, mcdc)

        # Surface crossing
        if event == EVENT_SURFACE:
            kernel.surface_crossing(P, mcdc)

        # Mesh crossing
        elif event == EVENT_MESH:
            kernel.mesh_crossing(P, mcdc)

        # Surface and mesh crossing
        elif event == EVENT_SURFACE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.move_particle(P, -PRECISION)
            kernel.surface_crossing(P, mcdc)

        # Collision
        elif event == EVENT_COLLISION:
            # Get collision type
            event = kernel.collision(P, mcdc)

            # Perform collision
            if event == EVENT_CAPTURE:
                kernel.capture(P, mcdc)
            elif event == EVENT_SCATTERING:
                kernel.scattering(P, mcdc)
            elif event == EVENT_FISSION:
                kernel.fission(P, mcdc)
            elif event == EVENT_TIME_REACTION:
                kernel.time_reaction(P, mcdc)
        
        # Time boundary
        elif event == EVENT_TIME_BOUNDARY:
            kernel.time_boundary(P, mcdc)
    
        # Apply weight window
        if mcdc.setting.weight_window:
            mcdc.weight_window.apply_(P, mcdc)
