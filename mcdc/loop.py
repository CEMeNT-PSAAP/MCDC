import numpy as np
import numba
from numpy import ascontiguousarray as cga
from numba import njit, objmode, jit, uintp
from scipy.linalg import eig

import mcdc.kernel as kernel
import mcdc.type_  as type_
import mcdc.adapt  as adapt

import mcdc.print_ as print_module

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

    loop_index = numba.uint64(0)
    while not simulation_end:
        cycle_seed = kernel.int_hash_combo(loop_index, mcdc["rng_seed"])
        # Loop over source particles
        ls_seed = kernel.int_hash_combo(cycle_seed, 0x43616D696C6C65)
        loop_source(ls_seed, mcdc)

        # Loop over source precursors
        if mcdc["bank_precursor"]["size"] > 0:
            lsp_seed = kernel.int_hash_combo(cycle_seed, 0x546F6464)
            loop_source_precursor(lsp_seed, mcdc)

        # Eigenvalue cycle closeout
        if mcdc["setting"]["mode_eigenvalue"]:
            # Tally history closeout
            kernel.eigenvalue_tally_closeout_history(mcdc)
            if mcdc["cycle_active"]:
                kernel.tally_reduce_bin(mcdc)
                kernel.tally_closeout_history(mcdc)

            # Print progress
            with objmode():
                print_progress_eigenvalue(mcdc)

            # Manage particle banks
            mpb_seed = kernel.int_hash_combo(cycle_seed, 0x5279616E)
            kernel.manage_particle_banks(mpb_seed, mcdc)

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
            kernel.manage_particle_banks(cycle_seed, mcdc)

            # Increment census index
            mcdc["technique"]["census_idx"] += 1

        # Fixed-source closeout
        else:
            simulation_end = True

        loop_index += numba.uint64(1)

    # Tally closeout
    kernel.tally_closeout(mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        kernel.eigenvalue_tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(seed, mcdc):
    prog = mcdc
    # Rebase rng skip_ahead seed
    kernel.rng_skip_ahead_strides(mcdc["mpi_work_start"], mcdc)
    kernel.rng_rebase(mcdc)

    # Progress bar indicator
    N_prog = 0

    if mcdc["technique"]["iQMC"]:
        mcdc["technique"]["iqmc_sweep_counter"] += 1

    # Loop over particle sources
    for work_idx in range(mcdc["mpi_work_size"]):
        src_seed = kernel.int_hash_combo(numba.uint64(work_idx), seed)

        # Particle tracker
        if mcdc["setting"]["track_particle"]:
            mcdc["particle_track_history_ID"] += 1

        # =====================================================================
        # Get a source particle and put into active bank
        # =====================================================================

        # Get from fixed-source?
        if mcdc["bank_source"]["size"] == 0:
            P = np.zeros(1, dtype=type_.particle_record)[0]
            P["rng_seed"] = work_idx * 524287
            # Sample source
            sample_seed = kernel.int_hash_combo(src_seed, 0x496C68616D)
            xi = kernel.rng_from_seed(sample_seed)
            tot = 0.0
            for S in mcdc["sources"]:
                tot += S["prob"]
                if tot >= xi:
                    break
            P = kernel.source_particle(S, src_seed)

        # Get from source bank
        else:
            P = mcdc["bank_source"]["particles"][work_idx]

        # Check if it is beyond current census index
        census_idx = mcdc["technique"]["census_idx"]
        if P["t"] > mcdc["technique"]["census_time"][census_idx]:
            P["t"] += SHIFT
            #kernel.add_particle(P, mcdc["bank_census"])
            adapt.add_census(prog,P)
        else:
            # Add the source particle into the active bank
            #kernel.add_particle(P, mcdc["bank_active"])
            adapt.add_active(prog,P)

        # =====================================================================
        # Run the source particle and its secondaries
        # =====================================================================

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            # Get particle from active bank
            P = kernel.get_particle(mcdc["bank_active"], mcdc)

            # Apply weight window
            if mcdc["technique"]["weight_window"]:
                kernel.weight_window(P, prog)

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


# =========================================================================
# Particle loop
# =========================================================================


@njit
def loop_particle(P, prog):
    mcdc = adapt.device(prog)
    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)

    while P["alive"]:
        step_particle(P,prog)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)




def step_particle(P : adapt.particle, prog : uintp):
    mcdc = adapt.device(prog)

    # Find cell from root universe if unknown
    if P["cell_ID"] == -1:
        trans = np.zeros(3)
        P["cell_ID"] = kernel.get_particle_cell(P, 0, trans, mcdc)

    # Determine and move to event
    kernel.move_to_event(P, mcdc)
    event = P["event"]

    # The & operator here is a bitwise and.
    # It is used to determine if an event type is part of the particle event.

    # Collision
    if event & EVENT_COLLISION:
        # Generate IC?
        if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
            kernel.bank_IC(P, prog)

        # Branchless collision?
        if mcdc["technique"]["branchless_collision"]:
            kernel.branchless_collision(P, mcdc)

        # Analog collision
        else:
            # Get collision type
            kernel.collision(P, mcdc)
            event = P["event"]
            # Perform collision
            if event & EVENT_CAPTURE:
                kernel.capture(P, prog)
            elif event == EVENT_SCATTERING:
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
                    kernel.sensitivity_material(P, mcdc)

    # Mesh tally
    if event & EVENT_MESH:
        kernel.mesh_crossing(P, mcdc)

    # Different Methods for shifting the particle
    # Surface crossing
    if event & EVENT_SURFACE:
        kernel.surface_crossing(P, prog)

    # Surface move
    elif event & EVENT_SURFACE_MOVE:
        P["t"] += SHIFT
        P["cell_ID"] = -1

    # Time census
    elif event & EVENT_CENSUS:
        P["t"] += SHIFT
        adapt.add_census(prog,P)
        P["alive"] = False

    # Shift particle
    elif event & EVENT_LATTICE + EVENT_MESH:
        kernel.shift_particle(P, SHIFT)

    # Time boundary
    if event & EVENT_TIME_BOUNDARY:
        kernel.time_boundary(P, mcdc)

    # Apply weight window
    elif mcdc["technique"]["weight_window"]:
        kernel.weight_window(P, prog)

    # Apply weight roulette
    if mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P, mcdc)



# =============================================================================
# iQMC Loops
# =============================================================================


@njit
def loop_iqmc(prog):
    mcdc = adapt.device(prog)
    # generate material index
    kernel.generate_iqmc_material_idx(mcdc)
    # function calls from specified solvers
    if mcdc["setting"]["mode_eigenvalue"]:
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "davidson":
            davidson(prog)
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "power_iteration":
            power_iteration(prog)
    else:
        if mcdc["technique"]["iqmc_fixed_source_solver"] == "source_iteration":
            source_iteration(prog)
        if mcdc["technique"]["iqmc_fixed_source_solver"] == "gmres":
            gmres(prog)


@njit
def source_iteration(prog):
    simulation_end = False
    mcdc = adapt.device(prog)

    loop_index = numba.uint64(0)
    while not simulation_end:
        # reset particle bank size
        mcdc["bank_source"]["size"] = 0
        mcdc["technique"]["iqmc_source"] = np.zeros_like(
            mcdc["technique"]["iqmc_source"]
        )

        # set bank source
        kernel.prepare_qmc_source(mcdc)
        # initialize particles with LDS
        kernel.prepare_qmc_particles(prog)

        # prepare source for next iteration
        mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])

        # sweep particles
        loop_source(0, mcdc)
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

        # Print progress
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_iqmc(mcdc)

        # set flux_old = current flux
        mcdc["technique"]["iqmc_flux_old"] = mcdc["technique"]["iqmc_flux"].copy()

        loop_index += numba.uint64(1)


@njit
def gmres(prog):
    """
    GMRES solver for iQMC fixed-source problems.
    ----------

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html
    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html

    code adapted from: https://github.com/pygbe/pygbe/blob/master/pygbe/gmres.py

    """
    mcdc = adapt.device(prog)
    max_iter = mcdc["technique"]["iqmc_maxitt"]
    R = mcdc["technique"]["iqmc_krylov_restart"]
    tol = mcdc["technique"]["iqmc_tol"]
    vector_size = mcdc["technique"]["iqmc_flux"].size
    X = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), vector_size)
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

    # initial residual
    b = kernel.RHS(prog)
    r = b - kernel.AxV(X, b, prog)
    normr = np.linalg.norm(r)

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
            v[:] = kernel.AxV(vs[-1], b, prog)
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

                if rel_resid < tol:
                    break

            mcdc["technique"]["iqmc_itt"] += 1
            mcdc["technique"]["iqmc_res"] = rel_resid
            if not mcdc["setting"]["mode_eigenvalue"]:
                with objmode():
                    print_progress_iqmc(mcdc)
        # end inner loop, back to outer loop

        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = np.linalg.solve(H[0 : inner + 1, 0 : inner + 1].T, g[0 : inner + 1])
        update = np.ravel(np.dot(cga(V[: inner + 1, :].T), y.reshape(-1, 1)))
        X = X + update
        aux = kernel.AxV(X, b, prog)
        r = b - aux

        normr = np.linalg.norm(r)
        rel_resid = normr / res_0

        mcdc["technique"]["iqmc_itt"] += 1
        mcdc["technique"]["iqmc_res"] = rel_resid
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_iqmc(mcdc)

    # end outer loop


@njit
def power_iteration(prog):
    mcdc = adapt.device(prog)
    simulation_end = False

    # iteration tolerance
    tol = mcdc["technique"]["iqmc_tol"]
    # maximum number of iterations
    maxit = mcdc["technique"]["iqmc_maxitt"]
    mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()
    k_old = mcdc["k_eff"]
    solver = mcdc["technique"]["iqmc_fixed_source_solver"]

    while not simulation_end:
        # iterate over scattering source
        if solver == "source_iteration":
            source_iteration(prog)
        if solver == "gmres":
            gmres(prog)
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
        mcdc["technique"]["iqmc_res_outter"] = abs(mcdc["k_eff"] - k_old)
        k_old = mcdc["k_eff"]
        mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()
        mcdc["technique"]["iqmc_itt_outter"] += 1

        if mcdc["setting"]["progress_bar"]:
            with objmode():
                print_iqmc_eigenvalue_progress(mcdc)

        # iQMC convergence criteria
        if (mcdc["technique"]["iqmc_itt_outter"] == maxit) or (
            mcdc["technique"]["iqmc_res_outter"] <= tol
        ):
            simulation_end = True
    if mcdc["setting"]["progress_bar"]:
        with objmode():
            print_iqmc_eigenvalue_exit_code(mcdc)


@njit
def davidson(prog):
    """
    The generalized Davidson method is a Krylov subspace method for solving
    the generalized eigenvalue problem. The algorithm here is based on the
    outline in:

        Subramanian, C., et al. "The Davidson method as an alternative to
        power iterations for criticality calculations." Annals of nuclear
        energy 38.12 (2011): 2818-2823.

    """
    mcdc = adapt.device(prog)
    # TODO: handle imaginary eigenvalues

    # Davidson parameters
    simulation_end = False
    maxit = mcdc["technique"]["iqmc_maxitt"]
    tol = mcdc["technique"]["iqmc_tol"]
    # num_sweeps: number of preconditioner sweeps
    num_sweeps = mcdc["technique"]["iqmc_preconditioner_sweeps"]
    # m : restart parameter
    m = mcdc["technique"]["iqmc_krylov_restart"]
    k_old = mcdc["k_eff"]
    # initial size of Krylov subspace
    Vsize = 1
    # l : number of eigenvalues to solve for
    l = 1

    # initial scalar flux guess comes from power iteration
    mcdc["technique"]["iqmc_maxitt"] = 3
    mcdc["setting"]["progress_bar"] = False
    power_iteration(prog)
    mcdc["setting"]["progress_bar"] = True
    mcdc["technique"]["iqmc_maxitt"] = maxit
    mcdc["technique"]["iqmc_itt_outter"] = 0

    # resulting guess
    phi0 = mcdc["technique"]["iqmc_flux"].copy()
    Nt = phi0.size
    phi0 = np.reshape(phi0, (Nt,))

    # Krylov subspace matrices
    # we allocate memory then use slice indexing in loop
    V = np.zeros((Nt, maxit), dtype=np.float64)
    HV = np.zeros((Nt, maxit), dtype=np.float64)
    FV = np.zeros((Nt, maxit), dtype=np.float64)

    # orthonormalize initial guess
    V0 = phi0 / np.linalg.norm(phi0)
    V[:, 0] = V0

    if m is None:
        # unless specified there is no restart parameter
        m = maxit + 1

    # Davidson Routine
    while not simulation_end:
        # Calculate V*H*V (HxV is scattering linear operator function)
        HV[:, Vsize - 1] = kernel.HxV(V[:, :Vsize], prog)[:, 0]
        VHV = np.dot(cga(V[:, :Vsize].T), cga(HV[:, :Vsize]))
        # Calculate V*F*V (FxV is fission linear operator function)
        FV[:, Vsize - 1] = kernel.FxV(V[:, :Vsize], prog)[:, 0]
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
        res = kernel.FxV(u, prog) - Lambda * kernel.HxV(u, prog)
        mcdc["technique"]["iqmc_res_outter"] = abs(mcdc["k_eff"] - k_old)
        k_old = mcdc["k_eff"]
        mcdc["technique"]["iqmc_itt_outter"] += 1
        with objmode():
            print_iqmc_eigenvalue_progress(mcdc)

        # check convergence criteria
        if (mcdc["technique"]["iqmc_itt_outter"] == maxit) or (
            mcdc["technique"]["iqmc_res_outter"] <= tol
        ):
            simulation_end = True
            break
        else:
            # Precondition for next iteration
            t = kernel.preconditioner(res, prog, num_sweeps)
            # check restart condition
            if Vsize <= m - l:
                # appends new orthogonalization to V
                V[:, : Vsize + 1] = kernel.modified_gram_schmidt(V[:, :Vsize], t)
                Vsize += 1
            else:
                # "restarts" by appending to a new array
                Vsize = l + 1
                V[:, :Vsize] = kernel.modified_gram_schmidt(u, t)

    with objmode():
        print_iqmc_eigenvalue_exit_code(mcdc)

    # normalize and save final scalar flux
    flux = np.reshape(
        u / np.linalg.norm(u),
        mcdc["technique"]["iqmc_flux"].shape,
    )

    mcdc["technique"]["iqmc_flux"] = flux


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
        src_seed = kernel.int_hash_combo(numba.uint64(work_idx), seed)
        if kernel.rng_from_seed(src_seed) < w - N:
            N += 1
        DNP["w"] = N

        # =====================================================================
        # Loop over source particles from the source precursor
        # =====================================================================

        for particle_idx in range(N):
            # Create new particle
            P_new = np.zeros(1, dtype=type_.particle)[0]
            part_seed = kernel.int_hash_combo(particle_idx, src_seed)
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
                SigmaF = material["fission"][g]
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
                #kernel.add_particle(kernel.copy_particle(P_new), mcdc["bank_active"])
                adapt.add_active(prog,P_new)

                # Loop until active bank is exhausted
                while mcdc["bank_active"]["size"] > 0:
                    # Get particle from active bank
                    P = kernel.get_particle(mcdc["bank_active"], mcdc)

                    # Apply weight window
                    if mcdc["technique"]["weight_window"]:
                        kernel.weight_window(P, prog)

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
