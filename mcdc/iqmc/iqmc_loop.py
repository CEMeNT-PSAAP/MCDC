import numpy as np

from numpy import ascontiguousarray as cga
from numba import njit, objmode

import mcdc.adapt as adapt
import mcdc.iqmc.iqmc_kernel as iqmc_kernel
import mcdc.kernel as kernel
import mcdc.local as local

from mcdc.constant import *
from mcdc.loop import caching
from mcdc.print_ import (
    print_iqmc_eigenvalue_exit_code,
    print_iqmc_eigenvalue_progress,
    print_msg,
    print_progress,
    print_progress_iqmc,
)


@njit(cache=caching)
def iqmc_simulation(mcdc):
    # function calls from specified solvers
    iqmc = mcdc["technique"]["iqmc"]
    iqmc_kernel.iqmc_preprocess(mcdc)
    iqmc_kernel.lds_init(mcdc)

    if mcdc["setting"]["mode_eigenvalue"]:
        if iqmc["eigenmode_solver"] == "power_iteration":
            power_iteration(mcdc)
    else:
        if iqmc["fixed_source_solver"] == "source_iteration":
            source_iteration(mcdc)
        if iqmc["fixed_source_solver"] == "gmres":
            gmres(mcdc)


@njit(cache=caching)
def iqmc_loop_particle(P, prog):
    mcdc = adapt.device(prog)

    while P["alive"]:
        iqmc_step_particle(P, prog)


@njit(cache=caching)
def iqmc_step_particle(P, prog):
    mcdc = adapt.device(prog)

    # Determine and move to event
    iqmc_kernel.iqmc_move_to_event(P, mcdc)
    event = P["event"]

    # The & operator here is a bitwise and.
    # It is used to determine if an event type is part of the particle event.

    # Surface crossing
    if event & EVENT_SURFACE:
        kernel.surface_crossing(P, prog)
        if event & EVENT_DOMAIN:
            if not (
                mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_REFLECTIVE
                or mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_VACUUM
            ):
                kernel.domain_crossing(P, mcdc)

    # Lattice or mesh crossing (skipped if surface crossing)
    elif event & EVENT_LATTICE or event & EVENT_MESH:
        if event & EVENT_DOMAIN:
            kernel.domain_crossing(P, mcdc)

    # Moving surface transition
    if event & EVENT_SURFACE_MOVE:
        P["cell_ID"] = -1

    # Apply weight roulette
    if P["alive"] and mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P, mcdc)


@njit(cache=caching)
def iqmc_loop_source(mcdc):
    mcdc["technique"]["iqmc"]["sweep_counter"] += 1
    work_size = mcdc["bank_source"]["size"][0]
    N_prog = 0
    # loop over particles
    for idx_work in range(work_size):
        P = mcdc["bank_source"]["particles"][idx_work]
        mcdc["bank_source"]["size"] -= 1
        kernel.add_particle(P, mcdc["bank_active"])

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            P = local.particle()
            # Get particle from active bank
            kernel.get_particle(P, mcdc["bank_active"], mcdc)
            # Particle loop
            iqmc_loop_particle(P, mcdc)

        # Progress printout
        percent = (idx_work + 1.0) / work_size
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


@njit(cache=caching)
def source_iteration(mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    total_source_old = iqmc["total_source"].copy()

    while not simulation_end:
        # reset particle bank size
        kernel.set_bank_size(mcdc["bank_source"], 0)
        # initialize particles with LDS
        iqmc_kernel.iqmc_prepare_particles(mcdc)
        # reset tallies for next loop
        iqmc_kernel.iqmc_reset_tallies(iqmc)
        # sweep particles
        iqmc_loop_source(mcdc)

        # sum resultant flux on all processors
        iqmc_kernel.iqmc_distribute_tallies(iqmc)
        iqmc["itt"] += 1
        iqmc_kernel.iqmc_update_source(mcdc)
        # combine source tallies into one vector
        iqmc_kernel.iqmc_consolidate_sources(mcdc)
        # calculate norm of sources
        iqmc["res"] = iqmc_kernel.iqmc_res(iqmc["total_source"], total_source_old)
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


@njit(cache=caching)
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
    r = b - AxV(X, b, mcdc)
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
            v[:] = AxV(vs[-1], b, mcdc)
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
        aux = AxV(X, b, mcdc)
        r = b - aux
        normr = np.linalg.norm(r)
        rel_resid = normr / res_0
        iqmc["res"] = rel_resid
        if rel_resid < tol:
            break
        if iqmc["itt"] >= max_iter:
            return


# =============================================================================
# Linear operators
# =============================================================================


@njit(cache=caching)
def AxV(V, b, mcdc):
    """
    Linear operator to be used with GMRES.
    Calculate action of A on input vector V, where A is a transport sweep
    and V is the total source (constant and tilted).
    """
    iqmc = mcdc["technique"]["iqmc"]
    iqmc["total_source"] = V.copy()
    # distribute segments of V to appropriate sources
    iqmc_kernel.iqmc_distribute_sources(mcdc)
    # reset bank size
    kernel.set_bank_size(mcdc["bank_source"], 0)

    # QMC Sweep
    iqmc_kernel.iqmc_prepare_particles(mcdc)
    iqmc_kernel.iqmc_reset_tallies(iqmc)
    iqmc["sweep_counter"] += 1
    iqmc_loop_source(mcdc)
    # sum resultant flux on all processors
    iqmc_kernel.iqmc_distribute_tallies(iqmc)
    # update source adds effective scattering + fission + fixed-source
    iqmc_kernel.iqmc_update_source(mcdc)
    # combine all sources (constant and tilted) into one vector
    iqmc_kernel.iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()
    axv = V - (v_out - b)

    return axv
