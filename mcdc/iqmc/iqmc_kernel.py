import numpy as np

from mpi4py import MPI
from numpy import ascontiguousarray as cga
from numba import njit, objmode, literal_unroll

import mcdc.adapt as adapt
import mcdc.local as local
import mcdc.geometry as geometry

from mcdc.adapt import toggle, for_cpu, for_gpu
from mcdc.constant import *
from mcdc.kernel import (
    distance_to_boundary,
    distance_to_mesh,
    get_particle_material,
    mesh_get_index,
    move_particle,
)
from mcdc.loop import caching
from mcdc.type_ import iqmc_score_list

# =========================================================================
# Low-Discrepency Sequences
# =========================================================================


@toggle("iQMC")
def lds_init(mcdc):
    N, dim = mcdc["technique"]["iqmc"]["lds"].shape
    N_start = mcdc["mpi_work_start"]
    mcdc["technique"]["iqmc"]["lds"] = halton(N, dim, skip=N_start)


@toggle("iQMC")
def scramble_LDS(mcdc):
    # TODO: use MCDC seed system
    seed_batch = np.random.randint(1000, 1000000)
    N, dim = mcdc["technique"]["iqmc"]["lds"].shape
    N_start = mcdc["mpi_work_start"]
    mcdc["technique"]["iqmc"]["lds"] = rhalton(N, dim, seed=seed_batch, skip=N_start)


@toggle("iQMC")
def rhalton(N, dim, seed=12345, skip=0):
    np.random.seed(seed)
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19, 23, 29), dtype=np.int64)
    halton = np.zeros((N, dim), dtype=np.float64)

    for D in range(dim):
        b = primes[D]
        # b = np.int64(2)
        ind = np.arange(skip, skip + N, dtype=np.int64)
        b2r = 1 / b
        ans = np.zeros(ind.shape, dtype=np.float64)
        res = ind.copy()
        while (1.0 - b2r) < 1.0:
            dig = np.mod(res, b)
            perm = np.random.permutation(b)
            pdig = perm[dig]
            ans = ans + pdig.astype(np.float64) * b2r
            b2r = b2r / np.float64(b)
            res = ((res - dig) / b).astype(np.int64)
        halton[:, D] = ans

    return halton


@toggle("iQMC")
def halton(N, dim, skip=0):
    # TODO: find more efficient implementation of Halton Sequence
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19, 23, 29), dtype=np.int64)
    halton = np.zeros((N, dim), dtype=np.float64)

    for D in range(dim):
        b = primes[D]
        n, d = 0, 1
        for i in range(skip + N):
            x = d - n
            if x == 1:
                n = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n = (b + 1) * y - x
            if i >= skip:
                halton[i - skip, D] = n / d

    return halton


# =========================================================================
# Core iQMC functions
# =========================================================================


@toggle("iQMC")
def iqmc_move_to_event(P, mcdc):
    # =========================================================================
    # Get distances to events
    # =========================================================================

    # Distance to nearest geometry boundary (surface or lattice)
    # Also set particle material and speed
    d_boundary, event = distance_to_boundary(P, mcdc)

    # Distance to domain decomposition mesh
    d_domain = INF
    if mcdc["cycle_active"] and mcdc["technique"]["domain_decomposition"]:
        d_domain = distance_to_mesh(P, mcdc["technique"]["dd_mesh"], mcdc)

    # Distance to iqmc mesh
    d_mesh = distance_to_mesh(P, mcdc["technique"]["iqmc"]["mesh"], mcdc)

    # =========================================================================
    # Determine event
    #   Priority (in case of coincident events):
    #     boundary  > mesh
    # =========================================================================

    # Find the minimum
    distance = min(d_boundary, d_mesh, d_domain)

    # Remove the boundary event if it is not the nearest
    if d_boundary > distance * PREC:
        event = 0

    d_time_boundary = INF
    d_time_census = INF
    d_collision = INF
    # Add each event if it is within PREC of the nearest event
    if d_time_boundary <= distance * PREC:
        event += EVENT_TIME_BOUNDARY
    if d_time_census <= distance * PREC:
        event += EVENT_CENSUS
    if d_mesh <= distance * PREC:
        event += EVENT_MESH
    if d_domain <= distance * PREC:
        event += EVENT_DOMAIN
    if d_collision == distance:
        event = EVENT_COLLISION

    # Assign event
    P["event"] = event

    # =========================================================================
    # Move particle
    # =========================================================================

    # score iQMC tallies
    iqmc_score_tallies(P, distance, mcdc)
    # attenuate particle weight
    iqmc_continuous_weight_reduction(P, distance, mcdc)
    # kill particle if it falls below weight threshold
    if abs(P["w"]) <= mcdc["technique"]["iqmc"]["w_min"]:
        P["alive"] = False

    # Move particle
    move_particle(P, distance, mcdc)


@toggle("iQMC")
def iqmc_continuous_weight_reduction(P, distance, mcdc):
    """
    Continuous weight reduction technique based on particle track-length.
    """
    material = mcdc["materials"][P["material_ID"]]
    SigmaT = material["total"][:]
    w = P["iqmc"]["w"]
    P["iqmc"]["w"] = w * np.exp(-distance * SigmaT)
    P["w"] = P["iqmc"]["w"].sum()


@toggle("iQMC")
def iqmc_preprocess(mcdc):
    # set bank source
    iqmc = mcdc["technique"]["iqmc"]
    eigenmode = mcdc["setting"]["mode_eigenvalue"]
    # generate material index
    iqmc_generate_material_idx(mcdc)
    if iqmc["source"].all() == 0.0:
        # use material index to generate a first guess for the source
        iqmc_prepare_source(mcdc)
        iqmc_update_source(mcdc)
    if eigenmode and iqmc["eigenmode_solver"] == "power_iteration":
        iqmc_prepare_nuSigmaF(mcdc)

    iqmc_consolidate_sources(mcdc)


@toggle("iQMC")
def iqmc_prepare_nuSigmaF(mcdc):
    iqmc = mcdc["technique"]["iqmc"]
    mesh = iqmc["mesh"]
    flux = iqmc["score"]["flux"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # calculate nu*SigmaF for every cell
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    t = 0
                    mat_idx = iqmc["material_idx"][t, i, j, k]
                    material = mcdc["materials"][mat_idx]
                    iqmc["score"]["fission-source"] += iqmc_fission_source(
                        flux[:, t, i, j, k], material
                    )


@toggle("iQMC")
def iqmc_prepare_source(mcdc):
    """
    Iterates trhough all spatial cells to calculate the iQMC source. The source
    is a combination of the user input Fixed-Source plus the calculated
    Scattering-Source and Fission-Sources. Resutls are stored in
    mcdc['technique']['iqmc_source'], a matrix of size [G,Nt,Nx,Ny,Nz].

    """
    iqmc = mcdc["technique"]["iqmc"]
    flux_scatter = iqmc["score"]["flux"]
    flux_fission = iqmc["score"]["flux"]
    mesh = iqmc["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1

    fission = np.zeros_like(iqmc["source"])
    scatter = np.zeros_like(iqmc["source"])

    # calculate source for every cell and group in the iqmc_mesh
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    mat_idx = iqmc["material_idx"][t, i, j, k]
                    # we can vectorize the multigroup calculation here
                    fission[:, t, i, j, k] = iqmc_effective_fission(
                        flux_fission[:, t, i, j, k], mat_idx, mcdc
                    )
                    scatter[:, t, i, j, k] = iqmc_effective_scattering(
                        flux_scatter[:, t, i, j, k], mat_idx, mcdc
                    )
    iqmc["score"]["effective-scattering"] = scatter
    iqmc["score"]["effective-fission"] = fission
    iqmc["score"]["effective-fission-outter"] = fission
    iqmc_update_source(mcdc)


@toggle("iQMC")
def iqmc_prepare_particles(mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    """
    iqmc = mcdc["technique"]["iqmc"]
    # total number of particles
    N_particle = mcdc["setting"]["N_particle"]
    # number of particles this processor will handle
    N_work = mcdc["mpi_work_size"]

    # low discrepency sequence
    lds = iqmc["lds"]
    # source
    Q = iqmc["source"]
    mesh = iqmc["mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # total number of spatial cells
    N_total = Nx * Ny * Nz
    # outter mesh boundaries for sampling position
    xa = mesh["x"][0]
    xb = mesh["x"][-1]
    ya = mesh["y"][0]
    yb = mesh["y"][-1]
    za = mesh["z"][0]
    zb = mesh["z"][-1]

    for n in range(N_work):
        # Create new particle
        P_new = local.particle_record()
        # assign initial group, time, and rng_seed (not used)
        P_new["g"] = 0
        P_new["t"] = 0
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = iqmc_sample_position(xa, xb, lds[n, 0])
        P_new["y"] = iqmc_sample_position(ya, yb, lds[n, 4])
        P_new["z"] = iqmc_sample_position(za, zb, lds[n, 3])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = iqmc_sample_isotropic_direction(
            lds[n, 1], lds[n, 5]
        )
        t, x, y, z, outside = mesh_get_index(P_new, mesh)
        q = Q[:, t, x, y, z].copy()
        dV = iqmc_cell_volume(x, y, z, mesh)
        # Source tilt
        iqmc_tilt_source(t, x, y, z, P_new, q, mcdc)
        # set particle weight
        P_new["iqmc"]["w"] = q * dV * N_total / N_particle
        P_new["w"] = P_new["iqmc"]["w"].sum()
        # add to source bank
        adapt.add_source(P_new, mcdc)


@toggle("iQMC")
def iqmc_res(flux_new, flux_old):
    """
    Calculate residual between scalar flux iterations.

    """
    size = flux_old.size
    flux_new = np.linalg.norm(flux_new.reshape((size,)), ord=2)
    flux_old = np.linalg.norm(flux_old.reshape((size,)), ord=2)
    return (flux_new - flux_old) / flux_old


@toggle("iQMC")
def iqmc_score_tallies(P, distance, mcdc):
    """
    Tally the scalar flux and linear source tilt.

    """
    iqmc = mcdc["technique"]["iqmc"]
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    # Get indices
    mesh = iqmc["mesh"]
    material = mcdc["materials"][P["material_ID"]]
    w = P["iqmc"]["w"]
    SigmaT = material["total"]
    mat_id = P["material_ID"]

    t, x, y, z, outside = mesh_get_index(P, mesh)
    if outside:
        return

    dt = dx = dy = dz = 1.0
    if (mesh["t"][t] != -INF) and (mesh["t"][t] != INF):
        dt = mesh["t"][t + 1] - mesh["t"][t]
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]

    dV = dx * dy * dz * dt

    flux = iqmc_flux(SigmaT, w, distance, dV)
    score_bin["flux"][:, t, x, y, z] += flux

    # Score effective source tallies
    score_bin["effective-scattering"][:, t, x, y, z] += iqmc_effective_scattering(
        flux, mat_id, mcdc
    )
    score_bin["effective-fission"][:, t, x, y, z] += iqmc_effective_fission(
        flux, mat_id, mcdc
    )

    if score_list["fission-source"]:
        score_bin["fission-source"] += iqmc_fission_source(flux, material)

    if score_list["fission-power"]:
        score_bin["fission-power"][:, t, x, y, z] += iqmc_fission_power(flux, material)

    if score_list["tilt-x"]:
        x_mid = mesh["x"][x] + (dx * 0.5)
        tilt = iqmc_linear_tilt(P["ux"], P["x"], dx, x_mid, dy, dz, w, distance, SigmaT)
        score_bin["tilt-x"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)

    if score_list["tilt-y"]:
        y_mid = mesh["y"][y] + (dy * 0.5)
        tilt = iqmc_linear_tilt(P["uy"], P["y"], dy, y_mid, dx, dz, w, distance, SigmaT)
        score_bin["tilt-y"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)

    if score_list["tilt-z"]:
        z_mid = mesh["z"][z] + (dz * 0.5)
        tilt = iqmc_linear_tilt(P["uz"], P["z"], dz, z_mid, dx, dy, w, distance, SigmaT)
        score_bin["tilt-z"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)


@toggle("iQMC")
def iqmc_cell_volume(x, y, z, mesh):
    """
    Calculate the volume of the current spatial cell.

    """
    dx = dy = dz = 1
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]
    dV = dx * dy * dz
    return dV


@toggle("iQMC")
def iqmc_sample_position(a, b, sample):
    return a + (b - a) * sample


@toggle("iQMC")
def iqmc_sample_isotropic_direction(sample1, sample2):
    """
    Sample the an isotropic direction using samples between [0,1].

    """
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * sample1 - 1.0
    azi = 2.0 * PI * sample2

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    uy = math.cos(azi) * c
    uz = math.sin(azi) * c
    ux = mu
    return ux, uy, uz


@toggle("iQMC")
def iqmc_sample_group(sample, G):
    """
    Uniformly sample energy group using a random sample between [0,1].

    """
    return int(np.floor(sample * G))


@toggle("iQMC")
def iqmc_generate_material_idx(mcdc):
    """
    This algorithm is meant to loop through every spatial cell of the
    iQMC mesh and assign a material index according to the material_ID at
    the center of the cell.

    Therefore, the whole cell is treated as the material located at the
    center of the cell, regardless of whethere there are more materials
    present.

    A crude but quick approximation.
    """
    mesh = mcdc["technique"]["iqmc"]["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    dx = dy = dz = 1
    # create particle to utilize cell finding functions
    P_temp = local.particle()
    # set default attributes
    P_temp["alive"] = True
    P_temp["material_ID"] = -1
    P_temp["cell_ID"] = -1
    geometry.reset_local_coordinate(P_temp)

    x_mid = 0.5 * (mesh["x"][1:] + mesh["x"][:-1])
    y_mid = 0.5 * (mesh["y"][1:] + mesh["y"][:-1])
    z_mid = 0.5 * (mesh["z"][1:] + mesh["z"][:-1])

    # loop through every cell
    for t in range(Nt):
        for i in range(Nx):
            x = x_mid[i]
            for j in range(Ny):
                y = y_mid[j]
                for k in range(Nz):
                    z = z_mid[k]

                    # assign cell center position
                    P_temp["t"] = t
                    P_temp["x"] = x
                    P_temp["y"] = y
                    P_temp["z"] = z

                    # set cell_ID
                    P_temp["cell_ID"] = geometry.get_cell(P_temp, 0, mcdc)

                    # set material_ID
                    material_ID = get_particle_material(P_temp, mcdc)

                    # assign material index
                    mcdc["technique"]["iqmc"]["material_idx"][t, i, j, k] = material_ID


@toggle("iQMC")
def iqmc_reset_tallies(iqmc):
    score_bin = iqmc["score"]
    score_list = iqmc["score_list"]

    iqmc["source"].fill(0.0)
    for name in literal_unroll(iqmc_score_list):
        if score_list[name]:
            score_bin[name].fill(0.0)


@toggle("iQMC")
def iqmc_distribute_tallies(iqmc):
    score_bin = iqmc["score"]
    score_list = iqmc["score_list"]

    for name in literal_unroll(iqmc_score_list):
        if score_list[name]:
            iqmc_score_reduce_bin(score_bin[name])


@toggle("iQMC")
def iqmc_score_reduce_bin(score):
    # MPI Reduce
    buff = np.zeros_like(score)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array(score), buff, op=MPI.SUM)
    score[:] = buff


@toggle("iQMC")
def iqmc_update_source(mcdc):
    iqmc = mcdc["technique"]["iqmc"]
    keff = mcdc["k_eff"]
    scatter = iqmc["score"]["effective-scattering"]
    fixed = iqmc["fixed_source"]
    if (
        mcdc["setting"]["mode_eigenvalue"]
        and iqmc["eigenmode_solver"] == "power_iteration"
    ):
        fission = iqmc["score"]["effective-fission-outter"]
    else:
        fission = iqmc["score"]["effective-fission"]
    iqmc["source"] = scatter + (fission / keff) + fixed


@toggle("iQMC")
def iqmc_tilt_source(t, x, y, z, P, Q, mcdc):
    iqmc = mcdc["technique"]["iqmc"]
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    mesh = iqmc["mesh"]
    dx = mesh["x"][x + 1] - mesh["x"][x]
    dy = mesh["y"][y + 1] - mesh["y"][y]
    dz = mesh["z"][z + 1] - mesh["z"][z]
    x_mid = mesh["x"][x] + (0.5 * dx)
    y_mid = mesh["y"][y] + (0.5 * dy)
    z_mid = mesh["z"][z] + (0.5 * dz)
    # linear x-component
    if score_list["tilt-x"]:
        Q += score_bin["tilt-x"][:, t, x, y, z] * (P["x"] - x_mid)
    # linear y-component
    if score_list["tilt-y"]:
        Q += score_bin["tilt-y"][:, t, x, y, z] * (P["y"] - y_mid)
    # linear z-component
    if score_list["tilt-z"]:
        Q += score_bin["tilt-z"][:, t, x, y, z] * (P["z"] - z_mid)


@toggle("iQMC")
def iqmc_distribute_sources(mcdc):
    """
    This function is meant to distribute iqmc_total_source to the relevant
    invidual source contributions, e.x. source_total -> source, source-x,
    source-y, source-z, etc.

    """
    iqmc = mcdc["technique"]["iqmc"]
    total_source = iqmc["total_source"].copy()
    shape = iqmc["source"].shape
    size = iqmc["source"].size
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    Vsize = 0

    # effective source
    iqmc["source"] = np.reshape(total_source[Vsize : (Vsize + size)].copy(), shape)
    Vsize += size

    # source tilting arrays
    tilt_list = [
        "tilt-x",
        "tilt-y",
        "tilt-z",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            score_bin[name] = np.reshape(total_source[Vsize : (Vsize + size)], shape)
            Vsize += size


@toggle("iQMC")
def iqmc_consolidate_sources(mcdc):
    """
    This function is meant to collect the relevant invidual source
    contributions, e.x. source, source-x, source-y, source-z, source-xy, etc.
    and combine them into one vector (source_total)

    """
    iqmc = mcdc["technique"]["iqmc"]
    total_source = iqmc["total_source"]
    size = iqmc["source"].size
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    Vsize = 0

    # effective source
    total_source[Vsize : (Vsize + size)] = np.reshape(iqmc["source"].copy(), size)
    Vsize += size

    # source tilting arrays
    tilt_list = [
        "tilt-x",
        "tilt-y",
        "tilt-z",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            total_source[Vsize : (Vsize + size)] = np.reshape(score_bin[name], size)
            Vsize += size


# =============================================================================
# iQMC Tallies
# =============================================================================
# TODO: Not all ST tallies have been built for case where SigmaT = 0.0


@toggle("iQMC")
def iqmc_flux(SigmaT, w, distance, dV):
    # Score Flux
    if SigmaT.all() > 0.0:
        return w * (1 - np.exp(-(distance * SigmaT))) / (SigmaT * dV)
    else:
        return distance * w / dV


@toggle("iQMC")
def iqmc_fission_source(phi, material):
    SigmaF = material["fission"]
    nu_f = material["nu_f"]
    return np.sum(nu_f * SigmaF * phi)


@toggle("iQMC")
def iqmc_fission_power(phi, material):
    SigmaF = material["fission"]
    return SigmaF * phi


@toggle("iQMC")
def iqmc_effective_fission(phi, mat_id, mcdc):
    """
    Calculate the fission source for use with iQMC.

    """
    # TODO: Now, only single-nuclide material is allowed
    material = mcdc["nuclides"][mat_id]
    chi_p = material["chi_p"]
    chi_d = material["chi_d"]
    nu_p = material["nu_p"]
    nu_d = material["nu_d"]
    J = material["J"]
    SigmaF = material["fission"]
    F_p = np.dot(chi_p.T, nu_p * SigmaF * phi)
    F_d = np.dot(chi_d.T, (nu_d.T * SigmaF * phi).sum(axis=1))
    F = F_p + F_d

    return F


@toggle("iQMC")
def iqmc_effective_scattering(phi, mat_id, mcdc):
    """
    Calculate the scattering source for use with iQMC.

    """
    material = mcdc["materials"][mat_id]
    chi_s = material["chi_s"]
    SigmaS = material["scatter"]
    return np.dot(chi_s.T, SigmaS * phi)


@toggle("iQMC")
def iqmc_effective_source(phi, mat_id, mcdc):
    S = iqmc_effective_scattering(phi, mat_id, mcdc)
    F = iqmc_effective_fission(phi, mat_id, mcdc)
    return S + F


@toggle("iQMC")
def iqmc_linear_tilt(mu, x, dx, x_mid, dy, dz, w, distance, SigmaT):
    if SigmaT.all() > 1e-12:
        a = mu * (
            w * (1 - (1 + distance * SigmaT) * np.exp(-SigmaT * distance)) / SigmaT**2
        )
        b = (x - x_mid) * (w * (1 - np.exp(-SigmaT * distance)) / SigmaT)
        Q = 12 * (a + b) / (dx**3 * dy * dz)
    else:
        Q = mu * w * distance ** (2) / 2 + w * (x - x_mid) * distance
    return Q
