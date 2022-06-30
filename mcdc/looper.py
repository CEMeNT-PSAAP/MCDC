import numpy as np

from numba import njit, objmode

import mcdc.kernel as kernel

from mcdc.constant import *
from mcdc.print_   import print_progress, print_progress_eigenvalue

# =========================================================================
# Simulation loop
# =========================================================================

@njit
def loop_simulation(mcdc):
    # Distribute work to processors
    kernel.distribute_work(mcdc['setting']['N_hist'], mcdc)

    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)
        
        # Eigenvalue mode generation closeout
        if mcdc['setting']['mode_eigenvalue']:
            kernel.tally_closeout(mcdc)
            with objmode():
                print_progress_eigenvalue(mcdc)

        # Simulation end?
        if mcdc['setting']['mode_eigenvalue']:
            mcdc['i_iter'] += 1
            if mcdc['i_iter'] == mcdc['setting']['N_iter']: 
                simulation_end = True
        elif mcdc['bank_census']['size'] == 0: 
            simulation_end = True

        # Manage particle banks
        if not simulation_end:
            kernel.manage_particle_banks(mcdc)
            
    # Fixed-source mode closeout
    if not mcdc['setting']['mode_eigenvalue']:
        kernel.tally_closeout(mcdc)


# =========================================================================
# Source loop
# =========================================================================

@njit
def loop_source(mcdc):
    # Rebase rng skip_ahead seed
    kernel.rng_skip_ahead_strides(mcdc['mpi_work_start'], mcdc)
    kernel.rng_rebase(mcdc)

    # Progress bar indicator
    N_prog = 0
    
    # Loop over particle sources
    for work_idx in range(mcdc['mpi_work_size']):
        # Initialize RNG wrt work index
        kernel.rng_skip_ahead_strides(work_idx, mcdc)

        # Get a source particle and put into history bank
        if mcdc['bank_source']['size'] == 0:
            # Sample source
            xi  = kernel.rng(mcdc)
            tot = 0.0
            for S in mcdc['sources']:
                tot += S['prob']
                if tot >= xi:
                    break
            P = kernel.source_particle(S, mcdc)
            kernel.set_universe(P, mcdc)
        else:
            P = mcdc['bank_source']['particles'][work_idx]
        kernel.add_particle(P, mcdc['bank_history'])

        # Run the source particle and its secondaries
        # (until history bank is exhausted)
        while mcdc['bank_history']['size'] > 0:
            # Get particle from history bank
            P = kernel.pop_particle(mcdc['bank_history'])

            # Apply weight window
            if mcdc['technique']['weight_window']:
                kernel.weight_window(P, mcdc)
            
            # Particle loop
            loop_particle(P, mcdc)

        # Tally history closeout
        kernel.tally_closeout_history(mcdc)
        
        # Progress printout
        percent = (work_idx+1.0)/mcdc['mpi_work_size']
        if mcdc['setting']['progress_bar'] and int(percent*100.0) > N_prog:
            N_prog += 1
            with objmode(): 
                print_progress(percent)

        
# =========================================================================
# Particle loop
# =========================================================================

@njit
def loop_particle(P, mcdc):
    while P['alive']:
        # Determine and move to event
        event = kernel.move_to_event(P, mcdc)

        # Collision
        if event == EVENT_COLLISION:
            # Branchless collision?
            if mcdc['technique']['branchless_collision']:
                kernel.branchless_collision(P, mcdc)
            else:
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
        
        # Mesh crossing
        elif event == EVENT_MESH:
            kernel.mesh_crossing(P, mcdc)

        # Surface crossing
        elif event == EVENT_SURFACE:
            kernel.surface_crossing(P, mcdc)

        # Lattice crossing
        elif event == EVENT_LATTICE:
            kernel.lattice_crossing(P, mcdc)
    
        # Surface and mesh crossing
        elif event == EVENT_SURFACE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.move_particle(P, -PRECISION)
            kernel.surface_crossing(P, mcdc)

        # Lattice and mesh crossing
        elif event == EVENT_LATTICE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.move_particle(P, -PRECISION)
            kernel.lattice_crossing(P, mcdc)

        # Time boundary
        elif event == EVENT_TIME_BOUNDARY:
            kernel.time_boundary(P, mcdc)

        # Apply weight window
        if mcdc['technique']['weight_window']:
            kernel.weight_window(P, mcdc)
