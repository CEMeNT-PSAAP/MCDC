import numpy as np

from numba import njit, objmode

import mcdc.kernel as kernel

from mcdc.constant import *
from mcdc.print_   import print_progress, print_progress_eigenvalue

# =========================================================================
# Main loop
# =========================================================================

@njit
def loop_main(mcdc):
    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)
        
        # Eigenvalue cycle closeout
        if mcdc['setting']['mode_eigenvalue']:
            # Tally history closeout
            kernel.global_tally_closeout_history(mcdc)
            if mcdc['cycle_active']:
                kernel.tally_closeout_history(mcdc)

            # Print progress
            with objmode():
                print_progress_eigenvalue(mcdc)

            # Manage particle banks
            kernel.manage_particle_banks(mcdc)
            
            # Cycle management
            mcdc['i_cycle'] += 1
            if mcdc['i_cycle'] == mcdc['setting']['N_cycle']: 
                simulation_end = True
            elif mcdc['i_cycle'] >= mcdc['setting']['N_inactive']:
                mcdc['cycle_active'] = True

        # Time census closeout
        elif mcdc['technique']['time_census'] and \
             mcdc['technique']['census_idx'] < len(mcdc['technique']['census_time'])-1:
            # Manage particle banks
            kernel.manage_particle_banks(mcdc)

            # Increment census index
            mcdc['technique']['census_idx'] += 1

        # Fixed-source closeout
        else:
            simulation_end = True

    # Tally closeout
    kernel.tally_closeout(mcdc)    


# =============================================================================
# Source loop
# =============================================================================

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

        # =====================================================================
        # Get a source particle and put into active bank
        # =====================================================================

        # Get from fixed-source?
        if mcdc['bank_source']['size'] == 0:
            # Sample source
            xi  = kernel.rng(mcdc)
            tot = 0.0
            for S in mcdc['sources']:
                tot += S['prob']
                if tot >= xi:
                    break
            P = kernel.source_particle(S, mcdc)

        # Get from source bank
        else:
            P = mcdc['bank_source']['particles'][work_idx]

        # Check if it is beyond
        census_idx = mcdc['technique']['census_idx']
        if P['t'] > mcdc['technique']['census_time'][census_idx]:
            P['t'] += SHIFT
            kernel.add_particle(P, mcdc['bank_census'])
        else:
            # Add the source particle into the active bank
            kernel.add_particle(P, mcdc['bank_active'])

        # =====================================================================
        # Run the source particle and its secondaries
        # =====================================================================

        # Loop until active bank is exhausted
        while mcdc['bank_active']['size'] > 0:
            # Get particle from active bank
            P = kernel.get_particle(mcdc['bank_active'])

            # Apply weight window
            if mcdc['technique']['weight_window']:
                kernel.weight_window(P, mcdc)
            
            # Particle loop
            loop_particle(P, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        # Tally history closeout for fixed-source simulation
        if not mcdc['setting']['mode_eigenvalue']:
            kernel.tally_closeout_history(mcdc)
        
        # Progress printout
        percent = (work_idx+1.0)/mcdc['mpi_work_size']
        if mcdc['setting']['progress_bar'] and int(percent*100.0) > N_prog:
            N_prog += 1
            with objmode(): 
                print_progress(percent, mcdc)

        
# =========================================================================
# Particle loop
# =========================================================================

@njit
def loop_particle(P, mcdc):
    while P['alive']:
        # Find cell from root universe if unknown
        if P['cell_ID'] == -1:
            trans        = np.zeros(3)
            P['cell_ID'] = kernel.get_particle_cell(P, 0, trans, mcdc)

        # Determine and move to event
        kernel.move_to_event(P, mcdc)
        event = P['event']

        # Collision
        if event == EVENT_COLLISION:
            # Generate IC?
            if mcdc['technique']['IC_generator'] and mcdc['cycle_active']:
                kernel.bank_IC(P, mcdc)

            # Branchless collision?
            if mcdc['technique']['branchless_collision']:
                kernel.branchless_collision(P, mcdc)

            # Analog collision
            else:
                # Get collision type
                kernel.collision(P, mcdc)
                event = P['event']

                # Perform collision
                if event == EVENT_CAPTURE:
                    kernel.capture(P, mcdc)
                elif event == EVENT_SCATTERING:
                    kernel.scattering(P, mcdc)
                elif event == EVENT_FISSION:
                    kernel.fission(P, mcdc)

        # Mesh crossing
        elif event == EVENT_MESH:
            kernel.mesh_crossing(P, mcdc)

        # Surface crossing
        elif event == EVENT_SURFACE:
            kernel.surface_crossing(P, mcdc)

        # Lattice crossing
        elif event == EVENT_LATTICE:
            kernel.shift_particle(P, SHIFT)
    
        # Time boundary
        elif event == EVENT_TIME_BOUNDARY:
            kernel.mesh_crossing(P, mcdc)
            kernel.time_boundary(P, mcdc)

        # Surface move
        elif event == EVENT_SURFACE_MOVE:
            P['t']       += SHIFT
            P['cell_ID']  = -1

        # Time census
        elif event == EVENT_CENSUS:
            P['t'] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc['bank_census'])
            P['alive'] = False

        # Surface and mesh crossing
        elif event == EVENT_SURFACE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.surface_crossing(P, mcdc)

        # Lattice and mesh crossing
        elif event == EVENT_LATTICE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.shift_particle(P, SHIFT)

        # Surface move and mesh crossing
        elif event == EVENT_SURFACE_MOVE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            P['t']       += SHIFT
            P['cell_ID']  = -1

        # Time census and mesh crossing
        elif event == EVENT_CENSUS_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            P['t'] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc['bank_census'])
            P['alive'] = False

        # Apply weight window
        if mcdc['technique']['weight_window']:
            kernel.weight_window(P, mcdc)
