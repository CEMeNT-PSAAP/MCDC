from   mcdc.constant import *
import mcdc.kernel   as     kernel
from   mcdc.print_   import print_progress

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

# =========================================================================
# Source loop
# =========================================================================

def loop_source():
    # Loop over particle sources
    for work_idx in range(mcdc.mpi.work_size):
        # Initialize RNG wrt work index
        mcdc.rng.skip_ahead(work_idx)

        # Get a source particle and put into history bank
        if not mcdc.bank_source:
            P = mcdc.source.get_particle()
            kernel.set_cell(P)
            kernel.set_census_time_idx(P)
        else:
            P = mcdc.bank_source[work_idx]
        mcdc.bank_history.append(P)

        # Apply weight window
        if mcdc.weight_window is not None:
            mcdc.weight_window(P, mcdc.bank_history)

        # Run particle source and secondaries
        while mcdc.bank_history:
            # Get particle from history bank
            P = mcdc.bank_history.pop()
            
            # Particle loop
            loop_particle(P)

        # Tally history closeout
        mcdc.tally.closeout_history()
        
        # Progress printout
        print_progress(work_idx)

        
# =========================================================================
# Particle loop
# =========================================================================

def loop_particle(P):
    while P.alive:
        # Determine and move to event
        event = kernel.move_to_event(P)

        # Perform event
        if event != EVENT_MESH:
            # Surface crossing
            if event == EVENT_SURFACE:
                kernel.surface_crossing(P)

            # Collision
            elif event == EVENT_COLLISION:
                # Get collision type
                event = kernel.collision(P)

                # Perform collision
                if event == EVENT_CAPTURE:
                    kernel.capture(P)
                elif event == EVENT_SCATTERING:
                    kernel.scattering(P)
                elif event == EVENT_FISSION:
                    kernel.fission(P)
                elif event == EVENT_TIME_REACTION:
                    kernel.time_reaction(P)

            # Time census
            elif event == EVENT_CENSUS:
                kernel.event_census(P)
        
        # Apply weight window
        if P.alive and mcdc.weight_window is not None:
            kernel.weight_window(P, mcdc.bank_history)
