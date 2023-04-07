import numpy as np

from numba import njit, objmode

import mcdc.kernel as kernel
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import (
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
)


# =========================================================================
# Main loop
# =========================================================================


@njit
def loop_main(mcdc):
    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)

        # Loop over source precursors
        if mcdc["bank_precursor"]["size"] > 0:
            loop_source_precursor(mcdc)

        # Eigenvalue cycle closeout
        if mcdc["setting"]["mode_eigenvalue"]:
            # Tally history closeout
            kernel.eigenvalue_tally_closeout_history(mcdc)
            if mcdc["cycle_active"]:
                kernel.tally_closeout_history(mcdc)

            # Print progress
            with objmode():
                print_progress_eigenvalue(mcdc)

            # Manage particle banks
            kernel.manage_particle_banks(mcdc)

            # Cycle management
            mcdc["i_cycle"] += 1
            if mcdc["i_cycle"] == mcdc["setting"]["N_cycle"]:
                simulation_end = True
            elif mcdc["i_cycle"] >= mcdc["setting"]["N_inactive"]:
                mcdc["cycle_active"] = True

        # Time census closeout
        elif (
            mcdc["technique"]["time_census"]
            and mcdc["technique"]["census_idx"]
            < len(mcdc["technique"]["census_time"]) - 1
        ):
            # Manage particle banks
            kernel.manage_particle_banks(mcdc)

            # Increment census index
            mcdc["technique"]["census_idx"] += 1

        # Fixed-source closeout
        else:
            simulation_end = True

    # Tally closeout
    kernel.tally_closeout(mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        kernel.eigenvalue_tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(mcdc):
    # Rebase rng skip_ahead seed
    kernel.rng_skip_ahead_strides(mcdc["mpi_work_start"], mcdc)
    kernel.rng_rebase(mcdc)

    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    for work_idx in range(mcdc["mpi_work_size"]):
        # Particle tracker
        if mcdc["setting"]["track_particle"]:
            mcdc["particle_track_history_ID"] += 1

        # Initialize RNG wrt work index
        kernel.rng_skip_ahead_strides(work_idx, mcdc)

        # =====================================================================
        # Get a source particle and put into active bank
        # =====================================================================

        # Get from fixed-source?
        if mcdc["bank_source"]["size"] == 0:
            # Sample source
            xi = kernel.rng(mcdc)
            tot = 0.0
            for S in mcdc["sources"]:
                tot += S["prob"]
                if tot >= xi:
                    break
            P = kernel.source_particle(S, mcdc)

        # Get from source bank
        else:
            P = mcdc["bank_source"]["particles"][work_idx]

        # Check if it is beyond current census index
        census_idx = mcdc["technique"]["census_idx"]
        if P["t"] > mcdc["technique"]["census_time"][census_idx]:
            P["t"] += SHIFT
            kernel.add_particle(P, mcdc["bank_census"])
        else:
            # Add the source particle into the active bank
            kernel.add_particle(P, mcdc["bank_active"])

        # =====================================================================
        # Run the source particle and its secondaries
        # =====================================================================

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            # Get particle from active bank
            P = kernel.get_particle(mcdc["bank_active"], mcdc)

            # Apply weight window
            if mcdc["technique"]["weight_window"]:
                kernel.weight_window(P, mcdc)

            # Particle tracker
            if mcdc["setting"]["track_particle"]:
                mcdc["particle_track_particle_ID"] += 1

            # Particle loop
            loop_particle(P, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        # Tally history closeout for fixed-source simulation
        if not mcdc["setting"]["mode_eigenvalue"]:
            kernel.tally_closeout_history(mcdc)

        # Progress printout
        percent = (work_idx + 1.0) / mcdc["mpi_work_size"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)

    # Re-sync RNG
    skip = mcdc["mpi_work_size_total"] - mcdc["mpi_work_start"]
    kernel.rng_skip_ahead_strides(skip, mcdc)
    kernel.rng_rebase(mcdc)


# =========================================================================
# Particle loop
# =========================================================================


@njit
def loop_particle(P, mcdc):
    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)

    while P["alive"]:
        # Find cell from root universe if unknown
        if P["cell_ID"] == -1:
            trans = np.zeros(3)
            P["cell_ID"] = kernel.get_particle_cell(P, 0, trans, mcdc)

        # Determine and move to event
        kernel.move_to_event(P, mcdc)
        event = P["event"]

        # Collision
        if event == EVENT_COLLISION:
            # Generate IC?
            if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
                kernel.bank_IC(P, mcdc)

            # Branchless collision?
            if mcdc["technique"]["branchless_collision"]:
                kernel.branchless_collision(P, mcdc)

            # Analog collision
            else:
                # Get collision type
                kernel.collision(P, mcdc)
                event = P["event"]

                # Perform collision
                if event == EVENT_CAPTURE:
                    kernel.capture(P, mcdc)
                elif event == EVENT_SCATTERING:
                    kernel.scattering(P, mcdc)
                elif event == EVENT_FISSION:
                    kernel.fission(P, mcdc)

                # Sensitivity quantification for nuclide?
                material = mcdc["materials"][P["material_ID"]]
                if material["sensitivity"] and P["sensitivity_ID"] == 0:
                    kernel.sensitivity_material(P, mcdc)

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
            P["t"] += SHIFT
            P["cell_ID"] = -1

        # Time census
        elif event == EVENT_CENSUS:
            P["t"] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc["bank_census"])
            P["alive"] = False

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
            P["t"] += SHIFT
            P["cell_ID"] = -1

        # Time census and mesh crossing
        elif event == EVENT_CENSUS_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            P["t"] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc["bank_census"])
            P["alive"] = False

        # Apply weight window
        if mcdc["technique"]["weight_window"]:
            kernel.weight_window(P, mcdc)

        # Apply weight roulette
        if mcdc["technique"]["weight_roulette"]:
            # check if weight has fallen below threshold
            if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
                kernel.weight_roulette(P, mcdc)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)


# =============================================================================
# iQMC Loop
# =============================================================================


@njit
def loop_iqmc(mcdc):
    # generate material index
    kernel.generate_iqmc_material_idx(mcdc)
    # function calls from specified solvers
    if mcdc["setting"]["mode_eigenvalue"]:
        power_iteration(mcdc)
    else:
        source_iteration(mcdc)


@njit
def source_iteration(mcdc):
    simulation_end = False

    while not simulation_end:
        # reset particle bank size
        mcdc["bank_source"]["size"] = 0
        mcdc["technique"]["iqmc_source"] = np.zeros_like(
            mcdc["technique"]["iqmc_source"]
        )

        # set bank source
        kernel.prepare_qmc_source(mcdc)
        # initialize particles with LDS
        kernel.prepare_qmc_particles(mcdc)

        # prepare source for next iteration
        mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])

        # sweep particles
        loop_source(mcdc)
        # sum resultant flux on all processors
        kernel.iqmc_distribute_flux(mcdc)
        mcdc["technique"]["iqmc_itt"] += 1

        # calculate norm of flux iterations
        mcdc["technique"]["iqmc_res"] = kernel.qmc_res(
            mcdc["technique"]["iqmc_flux"], mcdc["technique"]["iqmc_flux_old"]
        )

        # iQMC convergence criteria
        if (mcdc["technique"]["iqmc_itt"] == mcdc["technique"]["iqmc_maxitt"]) or (
            mcdc["technique"]["iqmc_res"] <= mcdc["technique"]["iqmc_tol"]
        ):
            simulation_end = True

        # Print progres
        if not mcdc["setting"]["mode_eigenvalue"]:
            print_progress_iqmc(mcdc)

        # set flux_old = current flux
        mcdc["technique"]["iqmc_flux_old"] = mcdc["technique"]["iqmc_flux"].copy()


@njit
def power_iteration(mcdc):
    simulation_end = False

    # iteration tolerance
    tol = mcdc["technique"]["iqmc_tol"]
    # maximum number of iterations
    maxit = mcdc["technique"]["iqmc_maxitt"]
    mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()

    # assign function call from specified solvers
    # inner_iteration = globals()[mcdc["technique"]["fixed_source_solver"]]

    while not simulation_end:
        # iterate over scattering source
        source_iteration(mcdc)
        # reset counter for inner iteration
        mcdc["technique"]["iqmc_itt"] = 0

        # update k_eff
        kernel.UpdateK(
            mcdc["k_eff"],
            mcdc["technique"]["iqmc_flux_outter"],
            mcdc["technique"]["iqmc_flux"],
            mcdc,
        )

        # calculate diff in flux
        mcdc["technique"]["iqmc_res_outter"] = kernel.qmc_res(
            mcdc["technique"]["iqmc_flux"], mcdc["technique"]["iqmc_flux_outter"]
        )
        mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()
        mcdc["technique"]["iqmc_itt_outter"] += 1

        print_iqmc_eigenvalue_progress(mcdc)

        # iQMC convergence criteria
        if (mcdc["technique"]["iqmc_itt_outter"] == maxit) or (
            mcdc["technique"]["iqmc_res_outter"] <= tol
        ):
            simulation_end = True

    print_iqmc_eigenvalue_exit_code(mcdc)


# =============================================================================
# Precursor source loop
# =============================================================================


@njit
def loop_source_precursor(mcdc):
    # TODO: censussed neutrons seeding is still not reproducible

    # Progress bar indicator
    N_prog = 0

    # =========================================================================
    # Sync. RNG skip ahead for reproducibility
    # =========================================================================

    # Exscan upper estimate of number of particles generated locally
    idx_start, N_local, N_global = kernel.bank_scanning_DNP(
        mcdc["bank_precursor"], mcdc
    )

    # Skip ahead and rebase
    kernel.rng_skip_ahead_strides(idx_start, mcdc)
    kernel.rng_rebase(mcdc)

    # =========================================================================
    # Loop over precursor sources
    # =========================================================================

    for work_idx in range(mcdc["mpi_work_size_precursor"]):
        # Get precursor
        DNP = mcdc["bank_precursor"]["precursors"][work_idx]

        # Set groups
        j = DNP["g"]
        g = DNP["n_g"]

        # Determine number of particles to be generated
        w = DNP["w"]
        N = math.floor(w)
        # "Roulette" the last particle
        if kernel.rng(mcdc) < w - N:
            N += 1
        DNP["w"] = N

        # =====================================================================
        # Loop over source particles from the source precursor
        # =====================================================================

        for particle_idx in range(N):
            # Create new particle
            P_new = np.zeros(1, dtype=type_.particle)[0]
            P_new["alive"] = True
            P_new["w"] = 1.0
            P_new["sensitivity_ID"] = 0

            # Set position
            P_new["x"] = DNP["x"]
            P_new["y"] = DNP["y"]
            P_new["z"] = DNP["z"]

            # Get material
            trans = np.zeros(3)
            P_new["cell_ID"] = kernel.get_particle_cell(P_new, 0, trans, mcdc)
            material_ID = kernel.get_particle_material(P_new, mcdc)
            material = mcdc["materials"][material_ID]
            G = material["G"]

            # Initialize RNG wrt particle current running index
            kernel.rng_skip_ahead_strides(particle_idx, mcdc)

            # Sample nuclide and get spectrum and decay constant
            N_nuclide = material["N_nuclide"]
            if N_nuclide == 1:
                nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
                spectrum = nuclide["chi_d"][j]
                decay = nuclide["decay"][j]
            else:
                SigmaF = material["fission"][g]
                nu_d = material["nu_d"][g]
                xi = kernel.rng(mcdc) * nu_d[j] * SigmaF
                tot = 0.0
                for i in range(N_nuclide):
                    nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
                    density = material["nuclide_densities"][i]
                    tot += density * nuclide["nu_d"][g, j] * nuclide["fission"][g]
                    if xi < tot:
                        # Nuclide determined, now get the constant and spectruum
                        spectrum = nuclide["chi_d"][j]
                        decay = nuclide["decay"][j]
                        break

            # Sample emission time
            P_new["t"] = -math.log(kernel.rng(mcdc)) / decay
            census_idx = mcdc["technique"]["census_idx"]
            if census_idx > 0:
                P_new["t"] += mcdc["technique"]["census_time"][census_idx - 1]

            # Accept if it is inside current census index
            if P_new["t"] < mcdc["technique"]["census_time"][census_idx]:
                # Reduce precursor weight
                DNP["w"] -= 1.0

                # Skip if it's beyond time boundary
                if P_new["t"] > mcdc["setting"]["time_boundary"]:
                    continue

                # Sample energy
                xi = kernel.rng(mcdc)
                tot = 0.0
                for g_out in range(G):
                    tot += spectrum[g_out]
                    if tot > xi:
                        break
                P_new["g"] = g_out

                # Sample direction
                (
                    P_new["ux"],
                    P_new["uy"],
                    P_new["uz"],
                ) = kernel.sample_isotropic_direction(mcdc)

                # Push to active bank
                kernel.add_particle(kernel.copy_particle(P_new), mcdc["bank_active"])

                # Loop until active bank is exhausted
                while mcdc["bank_active"]["size"] > 0:
                    # Get particle from active bank
                    P = kernel.get_particle(mcdc["bank_active"], mcdc)

                    # Apply weight window
                    if mcdc["technique"]["weight_window"]:
                        kernel.weight_window(P, mcdc)

                    # Particle tracker
                    if mcdc["setting"]["track_particle"]:
                        mcdc["particle_track_particle_ID"] += 1

                    # Particle loop
                    loop_particle(P, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        # Tally history closeout for fixed-source simulation
        if not mcdc["setting"]["mode_eigenvalue"]:
            kernel.tally_closeout_history(mcdc)

        # Progress printout
        percent = (work_idx + 1.0) / mcdc["mpi_work_size_precursor"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)

    # Re-sync RNG
    skip = N_global - idx_start
    kernel.rng_skip_ahead_strides(skip, mcdc)
    kernel.rng_rebase(mcdc)
