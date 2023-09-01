import numpy as np
from numpy import ascontiguousarray as cga
from numba import njit, objmode, jit, cuda
from scipy.linalg import eig

import mcdc.kernel as kernel
import mcdc.type_ as type_

import mcdc.print_ as print_module

from mcdc.constant import *
from mcdc.print_ import (
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
)

import mcdc.adapt as adapt
from mcdc.adapt import toggle, for_gpu, for_cpu, universal_arrays

# =========================================================================
# Main loop
# =========================================================================


@njit
def loop_main(mcdc):
    simulation_end = False

    idx_cycle = 0
    while not simulation_end:
        seed_cycle = kernel.split_seed(idx_cycle, mcdc["rng_seed"])

        # Loop over source particles
        seed_source = kernel.split_seed(seed_cycle, 0x43616D696C6C65)
        loop_source(seed_source, mcdc)

        # Loop over source precursors
        if kernel.get_bank_size(mcdc["bank_precursor"]) > 0:
            seed_source_precursor = kernel.split_seed(seed_cycle, 0x546F6464)
            loop_source_precursor(seed_source_precursor, mcdc)

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
            seed_bank = kernel.split_seed(seed_cycle, 0x5279616E)
            kernel.manage_particle_banks(seed_bank, mcdc)

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
            seed_bank = kernel.split_seed(seed_cycle, 0x5279616E)
            kernel.manage_particle_banks(seed_cycle, mcdc)

            # Increment census index
            mcdc["technique"]["census_idx"] += 1

        # Fixed-source closeout
        else:
            simulation_end = True

        idx_cycle += 1

    # Tally closeout
    kernel.tally_closeout(mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        kernel.eigenvalue_tally_closeout(mcdc)





# =========================================================================
# Particle loop
# =========================================================================

@njit
def step_particle(P, prog):

    mcdc = adapt.device(prog)

    # Find cell from root universe if unknown
    if P["cell_ID"] == -1:
        trans_struct = adapt.local_translate()
        trans = trans_struct["values"]
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
            if event & EVENT_CAPTURE:
                kernel.capture(P, mcdc)
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
                kernel.sensitivity_material(P, prog)

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
        adapt.add_census(kernel.copy_record(P), mcdc)
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

    # Particle tracker
    if mcdc["setting"]["track_particle"] and not P["alive"]:
        kernel.track_particle(P, mcdc)




# =============================================================================
# Source loop
# =============================================================================





@njit
def prep_particle(P,prog):

    mcdc = adapt.device(prog)

    # Apply weight window
    if mcdc["technique"]["weight_window"]:
        kernel.weight_window(P, prog)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.allocate_pid(P,mcdc)
    
    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)




@njit
def generate_source_particle(idx_work, seed, prog):

    mcdc = adapt.device(prog)

    seed_work = kernel.split_seed(idx_work, seed)

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
        kernel.allocate_hid(P,mcdc)

    # Check if it is beyond current census index
    census_idx = mcdc["technique"]["census_idx"]
    if P["t"] > mcdc["technique"]["census_time"][census_idx]:
        P["t"] += SHIFT
        adapt.add_census(P, mcdc)
    else:
        # Add the source particle into the active bank
        adapt.add_active(P, prog)



@njit
def exhaust_active_bank(mcdc):
    # Loop until active bank is exhausted
    P = adapt.local_particle()
    while kernel.get_particle(P, mcdc["bank_active"], mcdc):
        prep_particle(P,mcdc)
        while P["alive"]:
            step_particle(P,mcdc)
    kernel.set_bank_size(mcdc["bank_active"],0)


@for_cpu()
@njit
def process_sources(seed,mcdc):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    for idx_work in range(mcdc["mpi_work_size"]):
        generate_source_particle(idx_work,seed,mcdc)

        # Loop until active bank is exhausted
        exhaust_active_bank(mcdc)

        # Progress printout
        percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


source_gpu_rt = None



def make_gpu_process_sources(precursor):
    def inner(func):

        def work_maker(prog):
            pass

        if precursor:
            def make_work(prog: nb.uintp) -> nb.boolean:
                mcdc = adapt.device(prog)

                idx_work = adapt.global_add(mcdc["mpi_work_iter"],0,1)

                if idx_work >= mcdc["mpi_work_size"]:
                    return False
                
                generate_source_particle(nb.uint64(idx_work),mcdc["source_seed"],mcdc)
                return True
            work_maker = make_work
        else:
            def make_work(prog: nb.uintp) -> nb.boolean:
                mcdc = adapt.device(prog)

                idx_work = adapt.global_add(mcdc["mpi_work_iter"],0,1)

                if idx_work >= mcdc["mpi_work_size_precursor"]:
                    return False

                # Get precursor
                DNP = mcdc["bank_precursor"]["precursors"][idx_work]

                # Determine number of particles to be generated
                w = DNP["w"]
                N = math.floor(w)
                # "Roulette" the last particle
                seed_work = kernel.split_seed(nb.uint64(idx_work), mcdc["source_precursor_seed"])
                if kernel.rng_from_seed(seed_work) < w - N:
                    N += 1
                DNP["w"] = N

                for particle_idx in range(N):
                    generate_precursor_particle(DNP,particle_idx,seed_work,prog)

                return True
            work_maker = make_work



        def initialize(prog: nb.uintp):
            pass

        def finalize(prog: nb.uintp):
            pass

        def step(prog: nb.uintp, P: adapt.particle_gpu):
            mcdc = adapt.device(prog)
            if P["fresh"]:
                prep_particle(P,prog)
            P["fresh"] = False
            step_particle(P,prog)
            if P["alive"]:
                adapt.step_gpu(prog,P)

        async_fns = [step]
        base_fns = (initialize,finalize, work_maker)
        loop_spec = adapt.harm.RuntimeSpec("collaz",adapt.state_spec,base_fns,async_fns)

        
        global source_gpu_rt
        source_gpu_rt = loop_spec.harmonize_instance()

        @njit
        def gpu_process_sources(seed,mcdc):
            if(precursor):
                mcdc["source_seed"] = seed
            else:
                mcdc["source_precursor_seed"] = seed
            mcdc["mpi_work_iter"][0] = 0
            source_gpu_rt.store_state(mcdc)
            source_gpu_rt.init(4096)
            source_gpu_rt.exec(65536,288)
            mcdc = source_gpu_rt.load_state()[0]

        return gpu_process_sources
    return inner


@for_gpu(on_target=[make_gpu_process_sources(precursor=False)])
def process_sources(seed,mcdc):
    pass


@njit
def loop_source(seed, mcdc):

    if mcdc["technique"]["iQMC"]:
        mcdc["technique"]["iqmc_sweep_counter"] += 1

    process_sources(seed,mcdc)

    # =====================================================================
    # Closeout
    # =====================================================================

    # Tally history closeout for fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"]:
        kernel.tally_closeout_history(mcdc)


    # Re-sync RNG
    skip = mcdc["mpi_work_size_total"] - mcdc["mpi_work_start"]




# =============================================================================
# iQMC Loops
# =============================================================================

@toggle("iQMC")
def loop_iqmc(mcdc):
    # generate material index
    kernel.generate_iqmc_material_idx(mcdc)
    # function calls from specified solvers
    if mcdc["setting"]["mode_eigenvalue"]:
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "davidson":
            davidson(mcdc)
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "power_iteration":
            power_iteration(mcdc)
    else:
        if mcdc["technique"]["iqmc_fixed_source_solver"] == "source_iteration":
            source_iteration(mcdc)
        if mcdc["technique"]["iqmc_fixed_source_solver"] == "gmres":
            gmres(mcdc)


@njit
def source_iteration(mcdc):
    simulation_end = False

    loop_index = 0
    while not simulation_end:
        # reset particle bank size
        kernel.set_bank_size(mcdc["bank_source"],0)
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

        loop_index += 1


@toggle("iQMC")
def gmres(mcdc):
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
    b = kernel.RHS(mcdc)
    r = b - kernel.AxV(X, b, mcdc)
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
        aux = kernel.AxV(X, b, mcdc)
        r = b - aux

        normr = np.linalg.norm(r)
        rel_resid = normr / res_0

        mcdc["technique"]["iqmc_itt"] += 1
        mcdc["technique"]["iqmc_res"] = rel_resid
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_iqmc(mcdc)

    # end outer loop


@toggle("iQMC")
def power_iteration(mcdc):
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
            source_iteration(mcdc)
        if solver == "gmres":
            gmres(mcdc)
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


@toggle("iQMC")
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
    power_iteration(mcdc)
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
        HV[:, Vsize - 1] = kernel.HxV(V[:, :Vsize], mcdc)[:, 0]
        VHV = np.dot(cga(V[:, :Vsize].T), cga(HV[:, :Vsize]))
        # Calculate V*F*V (FxV is fission linear operator function)
        FV[:, Vsize - 1] = kernel.FxV(V[:, :Vsize], mcdc)[:, 0]
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
        if P_new["t"] <= mcdc["setting"]["time_boundary"]:
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
            adapt.add_active(kernel.copy_record(P_new), prog)


@for_cpu()
@njit
def process_source_precursors(seed,mcdc):

    # Progress bar indicator
    N_prog = 0

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
            generate_precursor_particle(DNP,particle_idx,seed_work,mcdc)
            # Loop until active bank is exhausted
            exhaust_active_bank(mcdc)
    
        # Progress printout
        percent = (idx_work + 1.0) / mcdc["mpi_work_size_precursor"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


@njit
def loop_source_precursor(seed, mcdc):
    # TODO: censussed neutrons seeding is still not reproducible


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

    process_source_precursors(seed,mcdc)

    # =====================================================================
    # Closeout
    # =====================================================================

    # Tally history closeout for fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"]:
        kernel.tally_closeout_history(mcdc)


    # Re-sync RNG
    skip = N_global - idx_start


@for_gpu(on_target=[make_gpu_process_sources(precursor=True)])
def process_source_precursors(seed,mcdc):
    pass

