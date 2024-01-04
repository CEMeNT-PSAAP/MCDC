import numpy as np
from numpy import ascontiguousarray as cga
from numba import njit, objmode, jit
from scipy.linalg import eig

import mcdc.kernel as kernel
import mcdc.type_ as type_

import mcdc.print_ as print_module

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
)


# =========================================================================
# Fixed-source loop
# =========================================================================


@njit
def loop_fixed_source(mcdc):
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
            loop_source(seed_source, mcdc)

            # Loop over source precursors
            if mcdc["bank_precursor"]["size"] > 0:
                seed_source_precursor = kernel.split_seed(
                    seed_census, SEED_SPLIT_SOURCE_PRECURSOR
                )
                loop_source_precursor(seed_source_precursor, mcdc)

            # Time census closeout
            if idx_census < mcdc["setting"]["N_census"] - 1:
                # TODO: Output tally (optional)

                # Manage particle banks: population control and work rebalance
                seed_bank = kernel.split_seed(seed_census, SEED_SPLIT_BANK)
                kernel.manage_particle_banks(seed_bank, mcdc)

        # Multi-batch closeout
        if mcdc["setting"]["N_batch"] > 1:
            # Tally history closeout
            kernel.tally_reduce_bin(mcdc)
            kernel.tally_closeout_history(mcdc)
            # Uq closeout
            if mcdc["technique"]["uq"]:
                kernel.uq_tally_closeout_batch(mcdc)

    # Tally closeout
    if mcdc["technique"]["uq"]:
        kernel.uq_tally_closeout(mcdc)
    else:
        kernel.tally_closeout(mcdc)


# =========================================================================
# Eigenvalue loop
# =========================================================================


@njit
def loop_eigenvalue(mcdc):
    # Loop over power iteration cycles
    for idx_cycle in range(mcdc["setting"]["N_cycle"]):
        seed_cycle = kernel.split_seed(idx_cycle, mcdc["setting"]["rng_seed"])

        # Loop over source particles
        seed_source = kernel.split_seed(seed_cycle, SEED_SPLIT_SOURCE)
        loop_source(seed_source, mcdc)

        # Tally "history" closeout
        kernel.eigenvalue_tally_closeout_history(mcdc)
        if mcdc["cycle_active"]:
            kernel.tally_reduce_bin(mcdc)
            kernel.tally_closeout_history(mcdc)

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
    kernel.tally_closeout(mcdc)
    kernel.eigenvalue_tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(seed, mcdc):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]
    work_end = work_start + work_size
    for idx_work in range(work_size):
        seed_work = kernel.split_seed(work_start + idx_work, seed)

        # Particle tracker
        if mcdc["setting"]["track_particle"]:
            mcdc["particle_track_history_ID"] += 1

        # =====================================================================
        # Get a source particle and put into active bank
        # =====================================================================

        # Get from fixed-source?
        if mcdc["bank_source"]["size"] == 0:
            # Sample source
            P = kernel.source_particle(seed_work, mcdc)

        # Get from source bank
        else:
            P = mcdc["bank_source"]["particles"][idx_work]

        # Check if it is beyond current census index
        idx_census = mcdc["idx_census"]
        if P["t"] > mcdc["setting"]["census_time"][idx_census]:
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

        # Tally history closeout for one-batch fixed-source simulation
        if not mcdc["setting"]["mode_eigenvalue"] and mcdc["setting"]["N_batch"] == 1:
            kernel.tally_closeout_history(mcdc)

        # Tally history closeout for multi-batch uq simulation
        if mcdc["technique"]["uq"]:
            kernel.uq_tally_closeout_history(mcdc)

        # Progress printout
        percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


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

        # The & operator here is a bitwise and.
        # It is used to determine if an event type is part of the particle event.

        # Collision events
        if event & EVENT_COLLISION:
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
                if event == EVENT_SCATTERING:
                    kernel.scattering(P, mcdc)
                elif event == EVENT_FISSION:
                    kernel.fission(P, mcdc)

                # Sensitivity quantification for nuclide?
                material = mcdc["materials"][P["material_ID"]]
                if material["sensitivity"] and (
                    P["sensitivity_ID"] == 0
                    or mcdc["technique"]["dsm_order"] == 2
                    and P["sensitivity_ID"] <= mcdc["setting"]["N_sensitivity"]
                ):
                    kernel.sensitivity_material(P, mcdc)

        # Surface crossing
        if event & EVENT_SURFACE:
            kernel.surface_crossing(P, mcdc)

        # Lattice or mesh crossing (skipped if surface crossing)
        elif event & EVENT_LATTICE or event & EVENT_MESH:
            kernel.shift_particle(P, SHIFT)

        # Moving surface transition
        if event & EVENT_SURFACE_MOVE:
            P["t"] += SHIFT
            P["cell_ID"] = -1

        # Census time crossing
        if event & EVENT_CENSUS:
            P["t"] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc["bank_census"])
            P["alive"] = False

        # Time boundary crossing
        if event & EVENT_TIME_BOUNDARY:
            P["alive"] = False

        # Apply weight window
        if P["alive"] and mcdc["technique"]["weight_window"]:
            kernel.weight_window(P, mcdc)

        # Apply weight roulette
        if P["alive"] and mcdc["technique"]["weight_roulette"]:
            # check if weight has fallen below threshold
            if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
                kernel.weight_roulette(P, mcdc)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)


# =============================================================================
# iQMC Loops
# =============================================================================


@njit
def loop_iqmc(mcdc):
    # function calls from specified solvers
    iqmc = mcdc["technique"]["iqmc"]
    kernel.iqmc_preprocess(mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        if iqmc["eigenmode_solver"] == "davidson":
            davidson(mcdc)
        if iqmc["eigenmode_solver"] == "power_iteration":
            power_iteration(mcdc)
    else:
        if iqmc["fixed_source_solver"] == "source_iteration":
            source_iteration(mcdc)
        if iqmc["fixed_source_solver"] == "gmres":
            gmres(mcdc)


@njit
def source_iteration(mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    total_source_old = iqmc["total_source"].copy()

    while not simulation_end:
        # reset particle bank size
        mcdc["bank_source"]["size"] = 0
        # initialize particles with LDS
        kernel.iqmc_prepare_particles(mcdc)
        # reset tallies for next loop
        kernel.iqmc_reset_tallies(iqmc)
        # sweep particles
        iqmc["sweep_counter"] += 1
        loop_source(0, mcdc)

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


@njit
def gmres(mcdc):
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
    r = b - kernel.AxV(X, b, mcdc)
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
            v[:] = kernel.AxV(vs[-1], b, mcdc)
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
        aux = kernel.AxV(X, b, mcdc)
        r = b - aux
        normr = np.linalg.norm(r)
        rel_resid = normr / res_0
        iqmc["res"] = rel_resid
        if rel_resid < tol:
            break
        if iqmc["itt"] >= max_iter:
            return


@njit
def power_iteration(mcdc):
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
            source_iteration(mcdc)
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


@njit
def davidson(mcdc):
    """
    The generalized Davidson method is a Krylov subspace method for solving
    the generalized eigenvalue problem. The algorithm here is based on the
    outline in:

        Subramanian, C., et al. "The Davidson method as an alternative to
        power iterations for criticality calculations." Annals of nuclear
        energy 38.12 (2011): 2818-2823.

    """
    # TODO: handle imaginary eigenvalues
    iqmc = mcdc["technique"]["iqmc"]
    # Davidson parameters
    simulation_end = False
    maxit = iqmc["maxitt"]
    tol = iqmc["tol"]
    # num_sweeps: number of preconditioner sweeps
    num_sweeps = iqmc["preconditioner_sweeps"]
    # m : restart parameter
    m = iqmc["krylov_restart"]
    k_old = mcdc["k_eff"]
    # initial size of Krylov subspace
    Vsize = 1
    # l : number of eigenvalues to solve for
    l = 1
    # vector size
    Nt = iqmc["total_source"].size
    # allocate memory then use slice indexing in loop
    V = np.zeros((Nt, m), dtype=np.float64)
    HV = np.zeros((Nt, m), dtype=np.float64)
    FV = np.zeros((Nt, m), dtype=np.float64)

    V0 = iqmc["total_source"].copy()
    V0 = kernel.preconditioner(V0, mcdc, num_sweeps=5)
    # orthonormalize initial guess
    V0 = V0 / np.linalg.norm(V0)
    V[:, 0] = V0

    if m is None:
        # unless specified there is no restart parameter
        m = maxit + 1

    # Davidson Routine
    while not simulation_end:
        # Calculate V*H*V (HxV is scattering linear operator function)
        HV[:, Vsize - 1] = kernel.HxV(V[:, :Vsize], mcdc)
        VHV = np.dot(cga(V[:, :Vsize].T), cga(HV[:, :Vsize]))
        # Calculate V*F*V (FxV is fission linear operator function)
        FV[:, Vsize - 1] = kernel.FxV(V[:, :Vsize], mcdc)
        VFV = np.dot(cga(V[:, :Vsize].T), cga(FV[:, :Vsize]))
        # solve for eigenvalues and vectors
        with objmode(Lambda="complex128[:]", w="complex128[:,:]"):
            Lambda, w = eig(VFV, b=VHV)
            Lambda = np.array(Lambda, dtype=np.complex128)
            w = np.array(w, dtype=np.complex128)

        assert Lambda.imag.all() == 0.0
        Lambda = Lambda.real
        w = w.real
        # get indices of eigenvalues from largest to smallest
        idx = np.flip(Lambda.argsort())
        # sort eigenvalues from largest to smallest
        Lambda = Lambda[idx]
        # take the l largest eigenvalues
        Lambda = Lambda[:l]
        # sort corresponding eigenvector (oriented by column)
        w = w[:, idx]
        # take the l largest eigenvectors
        w = w[:, :l]
        # assign keff
        mcdc["k_eff"] = Lambda[0]
        # Ritz vector
        u = np.dot(cga(V[:, :Vsize]), cga(w))
        # residual
        res = kernel.FxV(u, mcdc) - Lambda * kernel.HxV(u, mcdc)
        iqmc["res_outter"] = abs(mcdc["k_eff"] - k_old) / k_old
        k_old = mcdc["k_eff"]
        iqmc["itt_outter"] += 1
        with objmode():
            print_iqmc_eigenvalue_progress(mcdc)

        # check convergence criteria
        if (iqmc["itt_outter"] >= maxit) or (iqmc["res_outter"] <= tol):
            simulation_end = True
            with objmode():
                print_iqmc_eigenvalue_exit_code(mcdc)
            # return
        else:
            # Precondition for next iteration
            t = kernel.preconditioner(res, mcdc, num_sweeps)
            # check restart condition
            if Vsize <= m - l:
                # appends new orthogonalization to V
                V[:, : Vsize + 1] = kernel.modified_gram_schmidt(V[:, :Vsize], t)
                Vsize += 1
            else:
                # "restarts" by appending to a new array
                Vsize = l + 1
                V[:, :Vsize] = kernel.modified_gram_schmidt(u, t)
        # TODO: normalize and save final scalar flux
        iqmc["score"]["flux"] /= iqmc["score"]["flux"].sum()


# =============================================================================
# Precursor source loop
# =============================================================================


@njit
def loop_source_precursor(seed, mcdc):
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

        # Set groups
        j = DNP["g"]
        g = DNP["n_g"]

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
            # Create new particle
            P_new = np.zeros(1, dtype=type_.particle)[0]
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
            trans = np.zeros(3)
            P_new["cell_ID"] = kernel.get_particle_cell(P_new, 0, trans, mcdc)
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
                    continue

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
        percent = (idx_work + 1.0) / mcdc["mpi_work_size_precursor"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)
