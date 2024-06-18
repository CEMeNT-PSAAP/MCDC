import numpy as np
from numpy import ascontiguousarray as cga
from numba import njit, objmode, jit
from scipy.linalg import eig

from mpi4py import MPI

import mcdc.adapt as adapt
import mcdc.kernel as kernel
import mcdc.type_ as type_
import pathlib

import mcdc.print_ as print_module

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
    print_msg,
)

caching = True


def set_cache(setting):
    caching = setting

    if setting == False:
        print_msg(" Caching has been disabled")
        # p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')
        # p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')


# =============================================================================
# Functions for GPU Interop
# =============================================================================

# The symbols declared below will be overwritten to reference external code that
# manages GPU execution (if GPU execution is supported and selected)
alloc_state, free_state = [None] * 2

src_alloc_program, src_free_program, src_load_state, src_store_state = [None] * 4
src_init_program, src_exec_program, src_complete, src_clear_flags = [None] * 4

pre_alloc_program, pre_free_program, pre_load_state, pre_store_state = [None] * 4
pre_init_program, pre_exec_program, pre_complete, pre_clear_flags = [None] * 4


# If GPU execution is supported and selected, the functions shown below will
# be redefined to overwrite the above symbols and perform initialization/
# finalization of GPU state
@njit
def setup_gpu(mcdc):
    pass


@njit
def teardown_gpu(mcdc):
    pass


# =========================================================================
# Fixed-source loop
# =========================================================================

# about caching:
#     it is enabled as a default at the jit call level
#     to effectivly disable cache, delete the cache folder (often located in /MCDC/mcdc/__pycache__)
#     see more about cacheing here https://numba.readthedocs.io/en/stable/developer/caching.html


@njit(cache=caching)
def loop_fixed_source(data, mcdc):
    # Loop over batches
    for idx_batch in range(mcdc["setting"]["N_batch"]):
        mcdc["idx_batch"] = idx_batch
        seed_batch = kernel.split_seed(idx_batch, mcdc["setting"]["rng_seed"])

        # Print multi-batch header
        if mcdc["setting"]["N_batch"] > 1:
            with objmode():
                print_header_batch(mcdc)
            if mcdc["technique"]["uq"]:
                seed_uq = kernel.split_seed(seed_batch, SEED_SPLIT_UQ)
                kernel.uq_reset(mcdc, seed_uq)

        # Loop over time censuses
        for idx_census in range(mcdc["setting"]["N_census"]):
            mcdc["idx_census"] = idx_census
            seed_census = kernel.split_seed(seed_batch, SEED_SPLIT_CENSUS)

            # Loop over source particles
            seed_source = kernel.split_seed(seed_census, SEED_SPLIT_SOURCE)
            loop_source(seed_source, data, mcdc)

            # Loop over source precursors
            if kernel.get_bank_size(mcdc["bank_precursor"]) > 0:
                seed_source_precursor = kernel.split_seed(
                    seed_census, SEED_SPLIT_SOURCE_PRECURSOR
                )
                loop_source_precursor(seed_source_precursor, data, mcdc)

            # Time census closeout
            if idx_census < mcdc["setting"]["N_census"] - 1:
                # TODO: Output tally (optional)

                # Manage particle banks: population control and work rebalance
                seed_bank = kernel.split_seed(seed_census, SEED_SPLIT_BANK)
                kernel.manage_particle_banks(seed_bank, mcdc)

        # Multi-batch closeout
        if mcdc["setting"]["N_batch"] > 1:
            # Reset banks
            kernel.set_bank_size(mcdc["bank_source"], 0)
            kernel.set_bank_size(mcdc["bank_census"], 0)
            kernel.set_bank_size(mcdc["bank_active"], 0)

            # Tally history closeout
            kernel.tally_reduce(data, mcdc)
            kernel.tally_accumulate(data, mcdc)
            # Uq closeout
            if mcdc["technique"]["uq"]:
                kernel.uq_tally_closeout_batch(data, mcdc)

    # Tally closeout
    if mcdc["technique"]["uq"]:
        kernel.uq_tally_closeout(data, mcdc)
    kernel.tally_closeout(data, mcdc)


# =========================================================================
# Eigenvalue loop
# =========================================================================


@njit(cache=caching)
def loop_eigenvalue(data, mcdc):
    # Loop over power iteration cycles
    for idx_cycle in range(mcdc["setting"]["N_cycle"]):
        seed_cycle = kernel.split_seed(idx_cycle, mcdc["setting"]["rng_seed"])

        # Loop over source particles
        seed_source = kernel.split_seed(seed_cycle, SEED_SPLIT_SOURCE)
        loop_source(seed_source, data, mcdc)

        # Tally "history" closeout
        kernel.eigenvalue_tally_closeout_history(mcdc)
        if mcdc["cycle_active"]:
            kernel.tally_reduce(data, mcdc)
            kernel.tally_accumulate(data, mcdc)

        # Print progress
        with objmode():
            print_progress_eigenvalue(mcdc)

        # Manage particle banks
        seed_bank = kernel.split_seed(seed_cycle, SEED_SPLIT_BANK)
        kernel.manage_particle_banks(seed_bank, mcdc)

        # Entering active cycle?
        mcdc["idx_cycle"] += 1
        if mcdc["idx_cycle"] >= mcdc["setting"]["N_inactive"]:
            mcdc["cycle_active"] = True

    # Tally closeout
    kernel.tally_closeout(data, mcdc)
    kernel.eigenvalue_tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit(cache=caching)
def generate_source_particle(work_start, idx_work, seed, prog):
    mcdc = adapt.device(prog)

    seed_work = kernel.split_seed(work_start + idx_work, seed)

    # =====================================================================
    # Get a source particle and put into active bank
    # =====================================================================

    # Get from fixed-source?
    if kernel.get_bank_size(mcdc["bank_source"]) == 0:
        # Sample source
        P = kernel.source_particle(seed_work, mcdc)

    # Get from source bank
    else:
        P = mcdc["bank_source"]["particles"][idx_work]

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.allocate_hid(P, mcdc)

    # Check if it is beyond current census index
    idx_census = mcdc["idx_census"]
    if P["t"] > mcdc["setting"]["census_time"][idx_census]:
        if mcdc["technique"]["domain_decomposition"]:
            if mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]] > 0:
                P["w"] /= mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]]
            if kernel.particle_in_domain(P, mcdc):
                adapt.add_census(P, prog)
        else:
            adapt.add_census(P, prog)
    else:
        # Add the source particle into the active bank
        if mcdc["technique"]["domain_decomposition"]:
            if mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]] > 0:
                P["w"] /= mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]]
            if kernel.particle_in_domain(P, mcdc):
                adapt.add_active(P, prog)
        else:
            adapt.add_active(P, prog)


@njit(cache=caching)
def prep_particle(P, prog):
    mcdc = adapt.device(prog)

    # Apply weight window
    if mcdc["technique"]["weight_window"]:
        kernel.weight_window(P, prog)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.allocate_pid(P, mcdc)


@njit(cache=caching)
def exhaust_active_bank(data, prog):
    mcdc = adapt.device(prog)
    P = adapt.local_particle()
    # Loop until active bank is exhausted
    while kernel.get_bank_size(mcdc["bank_active"]) > 0:
        # Get particle from active bank
        kernel.get_particle(P, mcdc["bank_active"], mcdc)

        prep_particle(P, prog)

        # Particle loop
        loop_particle(P, data, mcdc)


@njit(cache=caching)
def source_closeout(prog, idx_work, N_prog, data):
    mcdc = adapt.device(prog)

    # Tally history closeout for one-batch fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"] and mcdc["setting"]["N_batch"] == 1:
        kernel.tally_accumulate(data, mcdc)

    # Tally history closeout for multi-batch uq simulation
    if mcdc["technique"]["uq"]:
        kernel.uq_tally_closeout_history(data, mcdc)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
    if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


@njit(cache=caching)
def source_dd_resolution(data, prog):
    mcdc = adapt.device(prog)

    kernel.dd_particle_send(mcdc)
    terminated = False
    max_work = 1
    kernel.dd_recv(mcdc)
    if mcdc["domain_decomp"]["work_done"]:
        terminated = True

    while not terminated:
        if kernel.get_bank_size(mcdc["bank_active"]) > 0:
            P = adapt.local_particle()
            # Loop until active bank is exhausted
            while kernel.get_bank_size(mcdc["bank_active"]) > 0:

                kernel.get_particle(P, mcdc["bank_active"], mcdc)
                if not kernel.particle_in_domain(P, mcdc) and P["alive"] == True:
                    print(f"recieved particle not in domain")

                # Apply weight window
                if mcdc["technique"]["weight_window"]:
                    kernel.weight_window(P, mcdc)

                # Particle tracker
                if mcdc["setting"]["track_particle"]:
                    mcdc["particle_track_particle_ID"] += 1

                # Particle loop
                loop_particle(P, data, mcdc)

                # Tally history closeout for one-batch fixed-source simulation
                if (
                    not mcdc["setting"]["mode_eigenvalue"]
                    and mcdc["setting"]["N_batch"] == 1
                ):
                    kernel.tally_accumulate(data, mcdc)

        # Send all domain particle banks
        kernel.dd_particle_send(mcdc)

        kernel.dd_recv(mcdc)

        # Progress printout
        """
        percent = 1 - work_remaining / max_work
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)
        """
        if kernel.dd_check_halt(mcdc):
            kernel.dd_check_out(mcdc)
            terminated = True


@njit
def loop_source(seed, data, mcdc):
    # Progress bar indicator
    N_prog = 0

    if mcdc["technique"]["domain_decomposition"]:
        kernel.dd_check_in(mcdc)

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]
    work_end = work_start + work_size

    for idx_work in range(work_size):

        # =====================================================================
        # Generate a source particle
        # =====================================================================

        generate_source_particle(work_start, idx_work, seed, mcdc)

        # =====================================================================
        # Run the source particle and its secondaries
        # =====================================================================

        exhaust_active_bank(data, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        source_closeout(mcdc, idx_work, N_prog, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


def gpu_sources_spec():

    def make_work(prog: nb.uintp) -> nb.boolean:
        mcdc = adapt.device(prog)

        idx_work = adapt.global_add(mcdc["mpi_work_iter"], 0, 1)

        if idx_work >= mcdc["mpi_work_size"]:
            return False

        generate_source_particle(
            mcdc["mpi_work_start"], nb.uint64(idx_work), mcdc["source_seed"], prog
        )
        return True

    def initialize(prog: nb.uintp):
        pass

    def finalize(prog: nb.uintp):
        pass

    base_fns = (initialize, finalize, make_work)

    def step(prog: nb.uintp, P: adapt.particle_gpu):
        mcdc = adapt.device(prog)
        if P["fresh"]:
            prep_particle(P, prog)
        P["fresh"] = False
        step_particle(P, data, prog)
        if P["alive"]:
            adapt.step_async(prog, P)

    async_fns = [step]
    return adapt.harm.RuntimeSpec("mcdc_source", adapt.state_spec, base_fns, async_fns)


@njit
def gpu_loop_source(seed, mcdc):
    # Progress bar indicator
    N_prog = 0

    # =====================================================================
    # GPU Interop
    # =====================================================================

    # Number of blocks to launch and number of iterations to run
    block_count = 240
    iter_count = 65536

    mcdc["mpi_work_iter"][0] = 0
    mcdc["source_seed"] = seed

    # Store the global state to the GPU
    src_store_state(mcdc["gpu_state"], mcdc)

    # Execute the program, and continue to do so until it is done
    src_exec_program(mcdc["source_program"], block_count, iter_count)
    while not src_complete(mcdc["source_program"]):
        src_exec_program(mcdc["source_program"], block_count, iter_count)

    # Recover the original program state
    src_load_state(mcdc, mcdc["gpu_state"])
    src_clear_flags(mcdc["source_program"])

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


# =========================================================================
# Particle loop
# =========================================================================


@njit(cache=caching)
def loop_particle(P, data, prog):
    mcdc = adapt.device(prog)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)

    while P["alive"]:
        step_particle(P, data, prog)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)


@njit(cache=caching)
def step_particle(P, data, prog):
    mcdc = adapt.device(prog)

    # Find cell from root universe if unknown
    if P["cell_ID"] == -1:
        trans_struct = adapt.local_translate()
        trans = trans_struct["values"]
        P["cell_ID"] = kernel.get_particle_cell(P, UNIVERSE_ROOT, trans, mcdc)

    # Determine and move to event
    kernel.move_to_event(P, data, mcdc)
    event = P["event"]

    # The & operator here is a bitwise and.
    # It is used to determine if an event type is part of the particle event.

    # Collision events
    if event & EVENT_COLLISION:
        # Generate IC?
        if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
            kernel.bank_IC(P, prog)

        # Branchless collision?
        if mcdc["technique"]["branchless_collision"]:
            kernel.branchless_collision(P, prog)

        # Analog collision
        else:
            # Get collision type
            kernel.collision(P, mcdc)
            event = P["event"]

            # Perform collision
            if event == EVENT_SCATTERING:
                kernel.scattering(P, prog)
            elif event == EVENT_FISSION:
                kernel.fission(P, prog)

            # Sensitivity quantification for nuclide?
            material = mcdc["materials"][P["material_ID"]]
            if material["sensitivity"] and (
                P["sensitivity_ID"] == 0
                or mcdc["technique"]["dsm_order"] == 2
                and P["sensitivity_ID"] <= mcdc["setting"]["N_sensitivity"]
            ):
                kernel.sensitivity_material(P, prog)

    # Surface crossing
    if event & EVENT_SURFACE:
        kernel.surface_crossing(P, data, prog)
        if event & EVENT_DOMAIN:
            if mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_NONE:
                kernel.domain_crossing(P, mcdc)

    # Lattice or mesh crossing (skipped if surface crossing)
    elif event & EVENT_LATTICE or event & EVENT_MESH:
        kernel.shift_particle(P, SHIFT)
        if event & EVENT_DOMAIN:
            kernel.domain_crossing(P, mcdc)

    # Moving surface transition
    if event & EVENT_SURFACE_MOVE:
        P["t"] += SHIFT
        P["cell_ID"] = -1

    # Census time crossing
    if event & EVENT_CENSUS:
        P["t"] += SHIFT
        adapt.add_census(P, prog)
        P["alive"] = False

    # Time boundary crossing
    if event & EVENT_TIME_BOUNDARY:
        P["alive"] = False

    # Apply weight window
    if P["alive"] and mcdc["technique"]["weight_window"]:
        kernel.weight_window(P, prog)

    # Apply weight roulette
    if P["alive"] and mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P, mcdc)


# =============================================================================
# iQMC Loops
# =============================================================================


@njit(cache=caching)
def loop_iqmc(data, mcdc):
    # function calls from specified solvers
    iqmc = mcdc["technique"]["iqmc"]
    kernel.iqmc_preprocess(mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        if iqmc["eigenmode_solver"] == "power_iteration":
            power_iteration(data, mcdc)
    else:
        if iqmc["fixed_source_solver"] == "source_iteration":
            source_iteration(data, mcdc)
        if iqmc["fixed_source_solver"] == "gmres":
            gmres(data, mcdc)


@njit(cache=caching)
def source_iteration(data, mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    total_source_old = iqmc["total_source"].copy()

    while not simulation_end:
        # reset particle bank size
        kernel.set_bank_size(mcdc["bank_source"], 0)
        # initialize particles with LDS
        kernel.iqmc_prepare_particles(mcdc)
        # reset tallies for next loop
        kernel.iqmc_reset_tallies(iqmc)
        # sweep particles
        iqmc["sweep_counter"] += 1
        loop_source(0, data, mcdc)

        # sum resultant flux on all processors
        kernel.iqmc_distribute_tallies(iqmc)
        iqmc["itt"] += 1
        kernel.iqmc_update_source(mcdc)
        # combine source tallies into one vector
        kernel.iqmc_consolidate_sources(mcdc)
        # calculate norm of sources
        iqmc["res"] = kernel.iqmc_res(iqmc["total_source"], total_source_old)
        # iQMC convergence criteria
        if (iqmc["itt"] == iqmc["maxitt"]) or (iqmc["res"] <= iqmc["tol"]):
            simulation_end = True

        # Print progress
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_iqmc(mcdc)

        # set  source_old = current source
        total_source_old = iqmc["total_source"].copy()


@njit(cache=caching)
def gmres(data, mcdc):
    """
    GMRES solver.
    ----------
    Linear Krylov solver. Solves problem of the form Ax = b.

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
        Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
        http://www-users.cs.umn.edu/~saad/books.html
    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html

    code adapted from: https://github.com/pygbe/pygbe/blob/master/pygbe/gmres.py

    """
    iqmc = mcdc["technique"]["iqmc"]
    max_iter = iqmc["maxitt"]
    R = iqmc["krylov_restart"]
    tol = iqmc["tol"]

    fixed_source = iqmc["fixed_source"]
    single_vector = iqmc["fixed_source"].size
    b = np.zeros_like(iqmc["total_source"])
    b[:single_vector] = np.reshape(fixed_source, fixed_source.size)
    X = iqmc["total_source"].copy()
    # initial residual
    r = b - kernel.AxV(X, b, data, mcdc)
    normr = np.linalg.norm(r)

    # Defining dimension
    dimen = X.size
    # Set number of outer and inner iterations
    if R > dimen:
        # set number of outter iterations to max allowable (A.shape[0])
        R = dimen
    max_inner = R
    xtype = np.float64

    # max_outer should be max_iter/max_inner but this might not be an integer
    # so we get the ceil of the division.
    # In the inner loop there is a if statement to break in case max_iter is
    # reached.
    max_outer = int(np.ceil(max_iter / max_inner))

    # Check initial guess ( scaling by b, if b != 0, must account for
    # case when norm(b) is very small)
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol * normb:
        return X, 0

    iteration = 0

    # GMRES starts here
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        Q = []
        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []
        V[0, :] = (1.0 / normr) * r
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb

        # RHS vector in the Krylov space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v = V[inner + 1, :]
            v[:] = kernel.AxV(vs[-1], b, data, mcdc)
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner + 1):
                vk = vs[k]
                alpha = np.dot(vk, v)
                H[inner, k] = alpha
                v[:] = vk * (-alpha) + v[:]

            normv = np.linalg.norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = (1.0 / H[inner, inner + 1]) * v

            # Apply for Givens rotations to H
            if inner > 0:
                for j in range(inner):
                    Qloc = Q[j]
                    H[inner, :][j : j + 2] = np.dot(Qloc, H[inner, :][j : j + 2])

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1
            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # Caclulate matrix rotations
                    c, s, _ = kernel.lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]], dtype=xtype)
                    Q.append(Qblock)

                    # Apply Givens Rotations to RHS for the linear system in
                    # the krylov space.
                    g[inner : inner + 2] = np.dot(Qblock, g[inner : inner + 2])

                    # Apply Givens rotations to H
                    H[inner, inner] = np.dot(Qblock[0, :], H[inner, inner : inner + 2])
                    H[inner, inner + 1] = 0.0

            iteration += 1

            if inner < max_inner - 1:
                normr = abs(g[inner + 1])
                rel_resid = normr / res_0
                iqmc["res"] = rel_resid

            iqmc["itt"] += 1
            if not mcdc["setting"]["mode_eigenvalue"]:
                with objmode():
                    print_progress_iqmc(mcdc)

            if rel_resid < tol:
                break
            if iqmc["itt"] >= max_iter:
                break

        # end inner loop, back to outer loop
        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = np.linalg.solve(H[0 : inner + 1, 0 : inner + 1].T, g[0 : inner + 1])
        update = np.ravel(np.dot(cga(V[: inner + 1, :].T), y.reshape(-1, 1)))
        X = X + update
        aux = kernel.AxV(X, b, data, mcdc)
        r = b - aux
        normr = np.linalg.norm(r)
        rel_resid = normr / res_0
        iqmc["res"] = rel_resid
        if rel_resid < tol:
            break
        if iqmc["itt"] >= max_iter:
            return


@njit(cache=caching)
def power_iteration(data, mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    # iteration tolerance
    tol = iqmc["tol"]
    # maximum number of iterations
    maxit = iqmc["maxitt"]
    score_bin = iqmc["score"]
    k_old = mcdc["k_eff"]
    solver = iqmc["fixed_source_solver"]

    fission_source_old = score_bin["fission-source"].copy()

    while not simulation_end:
        # iterate over scattering source
        if solver == "source_iteration":
            iqmc["maxitt"] = 1
            source_iteration(data, mcdc)
            iqmc["maxitt"] = maxit
        if solver == "gmres":
            gmres(mcdc)
        # reset counter for inner iteration
        iqmc["itt"] = 0

        # update k_eff
        mcdc["k_eff"] *= score_bin["fission-source"][0] / fission_source_old[0]

        # calculate diff in keff
        iqmc["res_outter"] = abs(mcdc["k_eff"] - k_old) / k_old
        k_old = mcdc["k_eff"]
        # store outter iteration values
        score_bin["effective-fission-outter"] = score_bin["effective-fission"].copy()
        fission_source_old = score_bin["fission-source"].copy()
        iqmc["itt_outter"] += 1

        with objmode():
            print_iqmc_eigenvalue_progress(mcdc)

        # iQMC convergence criteria
        if (iqmc["itt_outter"] == maxit) or (iqmc["res_outter"] <= tol):
            simulation_end = True
            with objmode():
                print_iqmc_eigenvalue_exit_code(mcdc)


# =============================================================================
# Precursor source loop
# =============================================================================


@njit(cache=caching)
def generate_precursor_particle(DNP, particle_idx, seed_work, prog):
    mcdc = adapt.device(prog)

    # Set groups
    j = DNP["g"]
    g = DNP["n_g"]

    # Create new particle
    P_new = adapt.local_particle()
    part_seed = kernel.split_seed(particle_idx, seed_work)
    P_new["rng_seed"] = part_seed
    P_new["alive"] = True
    P_new["w"] = 1.0
    P_new["sensitivity_ID"] = 0

    # Set position
    P_new["x"] = DNP["x"]
    P_new["y"] = DNP["y"]
    P_new["z"] = DNP["z"]

    # Get material
    trans_struct = adapt.local_translate()
    trans = trans_struct["values"]
    P_new["cell_ID"] = kernel.get_particle_cell(P_new, UNIVERSE_ROOT, trans, mcdc)
    material_ID = kernel.get_particle_material(P_new, mcdc)
    material = mcdc["materials"][material_ID]
    G = material["G"]

    # Sample nuclide and get spectrum and decay constant
    N_nuclide = material["N_nuclide"]
    if N_nuclide == 1:
        nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
        spectrum = nuclide["chi_d"][j]
        decay = nuclide["decay"][j]
    else:
        SigmaF = material["fission"][g]  # MG only
        nu_d = material["nu_d"][g]
        xi = kernel.rng(P_new) * nu_d[j] * SigmaF
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
    P_new["t"] = -math.log(kernel.rng(P_new)) / decay
    idx_census = mcdc["idx_census"]
    if idx_census > 0:
        P_new["t"] += mcdc["setting"]["census_time"][idx_census - 1]

    # Accept if it is inside current census index
    if P_new["t"] < mcdc["setting"]["census_time"][idx_census]:
        # Reduce precursor weight
        DNP["w"] -= 1.0

        # Skip if it's beyond time boundary
        if P_new["t"] > mcdc["setting"]["time_boundary"]:
            return

        # Sample energy
        xi = kernel.rng(P_new)
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
        ) = kernel.sample_isotropic_direction(P_new)

        # Push to active bank
        adapt.add_active(P_new, prog)


@njit(cache=caching)
def source_precursor_closeout(prog, idx_work, N_prog, data):
    mcdc = adapt.device(prog)

    # Tally history closeout for fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"]:
        kernel.tally_accumulate(data, mcdc)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size_precursor"]
    if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


@njit
def loop_source_precursor(seed, data, mcdc):
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

    # =========================================================================
    # Loop over precursor sources
    # =========================================================================

    for idx_work in range(mcdc["mpi_work_size_precursor"]):
        # Get precursor
        DNP = mcdc["bank_precursor"]["precursors"][idx_work]

        # Determine number of particles to be generated
        w = DNP["w"]
        N = math.floor(w)
        # "Roulette" the last particle
        seed_work = kernel.split_seed(idx_work, seed)
        if kernel.rng_from_seed(seed_work) < w - N:
            N += 1
        DNP["w"] = N

        # =====================================================================
        # Loop over source particles from the source precursor
        # =====================================================================

        for particle_idx in range(N):

            generate_precursor_particle(DNP, particle_idx, seed_work, mcdc)

            exhaust_active_bank(data, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        source_precursor_closeout(mcdc, idx_work, N_prog, data)


def gpu_precursor_spec():

    def make_work(prog: nb.uintp) -> nb.boolean:
        mcdc = adapt.device(prog)

        idx_work = adapt.global_add(mcdc["mpi_work_iter"], 0, 1)

        if idx_work >= mcdc["mpi_work_size_precursor"]:
            return False

        seed = mcdc["source_seed"]

        # Get precursor
        DNP = mcdc["bank_precursor"]["precursors"][idx_work]

        # Determine number of particles to be generated
        w = DNP["w"]
        N = math.floor(w)
        # "Roulette" the last particle
        seed_work = kernel.split_seed(idx_work, seed)
        if kernel.rng_from_seed(seed_work) < w - N:
            N += 1
        DNP["w"] = N

        # =====================================================================
        # Loop over source particles from the source precursor
        # =====================================================================

        for particle_idx in range(N):
            generate_precursor_particle(DNP, particle_idx, seed_work, prog)

        return True

    def initialize(prog: nb.uintp):
        pass

    def finalize(prog: nb.uintp):
        pass

    base_fns = (initialize, finalize, make_work)

    def step(prog: nb.uintp, P: adapt.particle_gpu):
        mcdc = adapt.device(prog)
        if P["fresh"]:
            prep_particle(P, prog)
        P["fresh"] = False
        step_particle(P, data, prog)
        if P["alive"]:
            adapt.step_async(prog, P)

    async_fns = [step]
    return adapt.harm.RuntimeSpec(
        "mcdc_precursor", adapt.state_spec, base_fns, async_fns
    )


@njit
def gpu_loop_source_precursor(seed, mcdc):
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

    # =====================================================================
    # GPU Interop
    # =====================================================================

    # Number of blocks to launch and number of iterations to run
    block_count = 240
    iter_count = 65536

    mcdc["mpi_work_iter"][0] = 0
    mcdc["source_seed"] = seed

    # Store the global state to the GPU
    pre_store_state(mcdc["gpu_state"], mcdc)

    # Execute the program, and continue to do so until it is done
    pre_exec_program(mcdc["source_program"], block_count, iter_count)
    while not pre_complete(mcdc["source_program"]):
        pre_exec_program(mcdc["source_program"], block_count, iter_count)

    # Recover the original program state
    pre_load_state(mcdc, mcdc["gpu_state"])
    pre_clear_flags(mcdc["source_program"])

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (moved out of loop)
    # =====================================================================

    source_precursor_closeout(mcdc, 1, 1, data)


def build_gpu_progs():

    src_spec = gpu_sources_spec()
    pre_spec = gpu_precursor_spec()

    adapt.harm.RuntimeSpec.bind_and_load()

    src_fns = src_spec.async_functions()
    pre_fns = pre_spec.async_functions()

    global alloc_state, free_state
    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

    global src_alloc_program, src_free_program, src_load_state, src_store_state
    global src_init_program, src_exec_program, src_complete, src_clear_flags
    src_alloc_program = src_fns["alloc_program"]
    src_free_program = src_fns["free_program"]
    src_load_state = src_fns["load_state"]
    src_store_state = src_fns["store_state"]
    src_init_program = src_fns["init_program"]
    src_exec_program = src_fns["exec_program"]
    src_complete = src_fns["complete"]
    src_clear_flags = src_fns["clear_flags"]

    global pre_alloc_program, pre_free_program, pre_load_state, pre_store_state
    global pre_init_program, pre_exec_program, pre_complete, pre_clear_flags
    pre_alloc_state = pre_fns["alloc_state"]
    pre_free_state = pre_fns["free_state"]
    pre_alloc_program = pre_fns["alloc_program"]
    pre_free_program = pre_fns["free_program"]
    pre_load_state = pre_fns["load_state"]
    pre_store_state = pre_fns["store_state"]
    pre_init_program = pre_fns["init_program"]
    pre_exec_program = pre_fns["exec_program"]
    pre_complete = pre_fns["complete"]
    pre_clear_flags = pre_fns["clear_flags"]

    @njit
    def real_setup_gpu(mcdc):
        arena_size = 0x10000
        block_count = 240
        mcdc["gpu_state"] = adapt.cast_voidptr_to_uintp(alloc_state())
        mcdc["source_program"] = adapt.cast_voidptr_to_uintp(
            src_alloc_program(mcdc["gpu_state"], arena_size)
        )
        src_init_program(mcdc["source_program"], 4096)
        mcdc["precursor_program"] = adapt.cast_voidptr_to_uintp(
            pre_alloc_program(mcdc["gpu_state"], arena_size)
        )
        pre_init_program(mcdc["precursor_program"], 4096)

    @njit
    def real_teardown_gpu(mcdc):
        src_free_program(adapt.cast_uintp_to_voidptr(mcdc["source_program"]))
        pre_free_program(adapt.cast_uintp_to_voidptr(mcdc["precursor_program"]))
        free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_state"]))

    global setup_gpu, teardown_gpu
    setup_gpu = real_setup_gpu
    teardown_gpu = real_teardown_gpu

    global loop_source, loop_source_precursor
    loop_source = gpu_loop_source
    loop_source_precursor = gpu_loop_source_precursor
