import math

from mpi4py import MPI
from numba import njit, objmode, literal_unroll
import numba

import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import print_error
from mcdc.type_ import score_list, iqmc_score_list
from mcdc.loop import loop_source


# =============================================================================
# Random sampling
# =============================================================================


@njit
def sample_isotropic_direction(P):
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * rng(P) - 1.0
    azi = 2.0 * PI * rng(P)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    y = math.cos(azi) * c
    z = math.sin(azi) * c
    x = mu
    return x, y, z


@njit
def sample_white_direction(nx, ny, nz, P):
    # Sample polar cosine
    mu = math.sqrt(rng(P))

    # Sample azimuthal direction
    azi = 2.0 * PI * rng(P)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu**2) ** 0.5

    if nz != 1.0:
        B = (1.0 - nz**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * nz * cos_azi - ny * sin_azi) * C
        y = ny * mu + (ny * nz * cos_azi + nx * sin_azi) * C
        z = nz * mu - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the formula
    else:
        B = (1.0 - ny**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * ny * cos_azi - nz * sin_azi) * C
        z = nz * mu + (nz * ny * cos_azi + nx * sin_azi) * C
        y = ny * mu - cos_azi * Ac * B
    return x, y, z


@njit
def sample_uniform(a, b, P):
    return a + rng(P) * (b - a)


# TODO: use cummulative density function and binary search
@njit
def sample_discrete(group, P):
    tot = 0.0
    xi = rng(P)
    for i in range(group.shape[0]):
        tot += group[i]
        if tot > xi:
            return i


@njit
def sample_piecewise_linear(cdf, P):
    xi = rng(P)

    # Get bin
    idx = binary_search(xi, cdf[1])

    # Linear interpolation
    x1 = cdf[1, idx]
    x2 = cdf[1, idx + 1]
    y1 = cdf[0, idx]
    y2 = cdf[0, idx + 1]
    return y1 + (xi - x1) * (y2 - y1) / (x2 - x1)


# =============================================================================
# Random number generator
#   LCG with hash seed-split
# =============================================================================


@njit(numba.uint64(numba.uint64, numba.uint64))
def wrapping_mul(a, b):
    return a * b


@njit(numba.uint64(numba.uint64, numba.uint64))
def wrapping_add(a, b):
    return a + b


def wrapping_mul_python(a, b):
    a = numba.uint64(a)
    b = numba.uint64(b)
    with np.errstate(all="ignore"):
        return a * b


def wrapping_add_python(a, b):
    a = numba.uint64(a)
    b = numba.uint64(b)
    with np.errstate(all="ignore"):
        return a + b


def adapt_rng(object_mode=False):
    global wrapping_add, wrapping_mul
    if object_mode:
        wrapping_add = wrapping_add_python
        wrapping_mul = wrapping_mul_python


@njit(numba.uint64(numba.uint64, numba.uint64))
def split_seed(key, seed):
    """murmur_hash64a"""
    multiplier = numba.uint64(0xC6A4A7935BD1E995)
    length = numba.uint64(8)
    rotator = numba.uint64(47)
    key = numba.uint64(key)
    seed = numba.uint64(seed)

    hash_value = numba.uint64(seed) ^ wrapping_mul(length, multiplier)

    key = wrapping_mul(key, multiplier)
    key ^= key >> rotator
    key = wrapping_mul(key, multiplier)
    hash_value ^= key
    hash_value = wrapping_mul(hash_value, multiplier)

    hash_value ^= hash_value >> rotator
    hash_value = wrapping_mul(hash_value, multiplier)
    hash_value ^= hash_value >> rotator
    return hash_value


@njit(numba.uint64(numba.uint64))
def rng_(seed):
    return wrapping_add(wrapping_mul(RNG_G, seed), RNG_C) & RNG_MOD_MASK


@njit
def rng(state):
    state["rng_seed"] = rng_(state["rng_seed"])
    return state["rng_seed"] / RNG_MOD


@njit
def rng_from_seed(seed):
    return rng_(seed) / RNG_MOD


@njit
def rng_array(seed, shape, size):
    xi = np.zeros(size)
    for i in range(size):
        xi_seed = split_seed(i, seed)
        xi[i] = rng_from_seed(xi_seed)
    xi = xi.reshape(shape)
    return xi


# =============================================================================
# Particle source operations
# =============================================================================


@njit
def source_particle(seed, mcdc):
    P = np.zeros(1, dtype=type_.particle_record)[0]
    P["rng_seed"] = seed

    # Sample source
    xi = rng(P)
    tot = 0.0
    for source in mcdc["sources"]:
        tot += source["prob"]
        if tot >= xi:
            break

    # Position
    if source["box"]:
        x = sample_uniform(source["box_x"][0], source["box_x"][1], P)
        y = sample_uniform(source["box_y"][0], source["box_y"][1], P)
        z = sample_uniform(source["box_z"][0], source["box_z"][1], P)
    else:
        x = source["x"]
        y = source["y"]
        z = source["z"]

    # Direction
    if source["isotropic"]:
        ux, uy, uz = sample_isotropic_direction(P)
    elif source["white"]:
        ux, uy, uz = sample_white_direction(
            source["white_x"], source["white_y"], source["white_z"], P
        )
    else:
        ux = source["ux"]
        uy = source["uy"]
        uz = source["uz"]

    # Energy and time
    if mcdc["setting"]["mode_MG"]:
        g = sample_discrete(source["group"], P)
        E = 0.0
    else:
        g = 0
        E = sample_piecewise_linear(source["energy"], P)

    # Time
    t = sample_uniform(source["time"][0], source["time"][1], P)

    # Make and return particle
    P["x"] = x
    P["y"] = y
    P["z"] = z
    P["t"] = t
    P["ux"] = ux
    P["uy"] = uy
    P["uz"] = uz
    P["g"] = g
    P["E"] = E
    P["w"] = 1.0

    P["sensitivity_ID"] = 0

    return P


# =============================================================================
# Particle bank operations
# =============================================================================


@njit
def add_particle(P, bank):
    # Check if bank is full
    if bank["size"] == bank["particles"].shape[0]:
        with objmode():
            print_error("Particle %s bank is full." % bank["tag"])

    # Set particle
    bank["particles"][bank["size"]] = P

    # Increment size
    bank["size"] += 1


@njit
def get_particle(bank, mcdc):
    # Check if bank is empty
    if bank["size"] == 0:
        with objmode():
            print_error("Particle %s bank is empty." % bank["tag"])

    # Decrement size
    bank["size"] -= 1

    # Create in-flight particle
    P = np.zeros(1, dtype=type_.particle)[0]

    # Set attribute
    P_rec = bank["particles"][bank["size"]]
    P["x"] = P_rec["x"]
    P["y"] = P_rec["y"]
    P["z"] = P_rec["z"]
    P["t"] = P_rec["t"]
    P["ux"] = P_rec["ux"]
    P["uy"] = P_rec["uy"]
    P["uz"] = P_rec["uz"]
    P["g"] = P_rec["g"]
    P["E"] = P_rec["E"]
    P["w"] = P_rec["w"]
    P["rng_seed"] = P_rec["rng_seed"]

    if mcdc["technique"]["iQMC"]:
        P["iqmc"]["w"] = P_rec["iqmc"]["w"]

    P["alive"] = True
    P["sensitivity_ID"] = P_rec["sensitivity_ID"]

    # Set default IDs and event
    P["material_ID"] = -1
    P["cell_ID"] = -1
    P["surface_ID"] = -1
    P["event"] = -1
    return P


@njit
def manage_particle_banks(seed, mcdc):
    # Record time
    if mcdc["mpi_master"]:
        with objmode(time_start="float64"):
            time_start = MPI.Wtime()

    if mcdc["setting"]["mode_eigenvalue"]:
        # Normalize weight
        normalize_weight(mcdc["bank_census"], mcdc["setting"]["N_particle"])

    # Population control
    if mcdc["technique"]["population_control"]:
        population_control(seed, mcdc)
    else:
        # Swap census and source bank
        size = mcdc["bank_census"]["size"]
        mcdc["bank_source"]["size"] = size
        mcdc["bank_source"]["particles"][:size] = mcdc["bank_census"]["particles"][
            :size
        ]

    # MPI rebalance
    bank_rebalance(mcdc)

    # Zero out census bank
    mcdc["bank_census"]["size"] = 0

    # Manage IC bank
    if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
        manage_IC_bank(mcdc)

    # Accumulate time
    if mcdc["mpi_master"]:
        with objmode(time_end="float64"):
            time_end = MPI.Wtime()
        mcdc["runtime_bank_management"] += time_end - time_start


@njit
def manage_IC_bank(mcdc):
    # Buffer bank
    buff_n = np.zeros(
        mcdc["technique"]["IC_bank_neutron_local"]["particles"].shape[0],
        dtype=type_.particle_record,
    )
    buff_p = np.zeros(
        mcdc["technique"]["IC_bank_precursor_local"]["precursors"].shape[0],
        dtype=type_.precursor,
    )

    with objmode(Nn="int64", Np="int64"):
        # Create MPI-supported numpy object
        Nn = mcdc["technique"]["IC_bank_neutron_local"]["size"]
        Np = mcdc["technique"]["IC_bank_precursor_local"]["size"]

        neutrons = MPI.COMM_WORLD.gather(
            mcdc["technique"]["IC_bank_neutron_local"]["particles"][:Nn]
        )
        precursors = MPI.COMM_WORLD.gather(
            mcdc["technique"]["IC_bank_precursor_local"]["precursors"][:Np]
        )

        if mcdc["mpi_master"]:
            neutrons = np.concatenate(neutrons[:])
            precursors = np.concatenate(precursors[:])

            # Set output buffer
            Nn = neutrons.shape[0]
            Np = precursors.shape[0]
            for i in range(Nn):
                buff_n[i] = neutrons[i]
            for i in range(Np):
                buff_p[i] = precursors[i]

    # Set global bank from buffer
    if mcdc["mpi_master"]:
        start_n = mcdc["technique"]["IC_bank_neutron"]["size"]
        start_p = mcdc["technique"]["IC_bank_precursor"]["size"]
        mcdc["technique"]["IC_bank_neutron"]["size"] += Nn
        mcdc["technique"]["IC_bank_precursor"]["size"] += Np
        for i in range(Nn):
            mcdc["technique"]["IC_bank_neutron"]["particles"][start_n + i] = buff_n[i]
        for i in range(Np):
            mcdc["technique"]["IC_bank_precursor"]["precursors"][start_p + i] = buff_p[
                i
            ]

    # Reset local banks
    mcdc["technique"]["IC_bank_neutron_local"]["size"] = 0
    mcdc["technique"]["IC_bank_precursor_local"]["size"] = 0


@njit
def bank_scanning(bank, mcdc):
    N_local = bank["size"]

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def bank_scanning_weight(bank, mcdc):
    # Local weight CDF
    N_local = bank["size"]
    w_cdf = np.zeros(N_local + 1)
    for i in range(N_local):
        w_cdf[i + 1] = w_cdf[i] + bank["particles"][i]["w"]
    W_local = w_cdf[-1]

    # Starting weight
    buff = np.zeros(1, dtype=np.float64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([W_local]), buff, MPI.SUM)
    w_start = buff[0]
    w_cdf += w_start

    # Global weight
    buff[0] = w_cdf[-1]
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    W_global = buff[0]

    return w_start, w_cdf, W_global


@njit
def bank_scanning_DNP(bank, mcdc):
    N_DNP_local = bank["size"]

    # Get sum of ceil-ed local DNP weights
    N_local = 0
    for i in range(N_DNP_local):
        DNP = bank["precursors"][i]
        N_local += math.ceil(DNP["w"])

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for P in bank["particles"]:
        P["w"] *= norm / W


@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(bank["size"]):
        W_local[0] += bank["particles"][i]["w"]

    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]


@njit
def bank_rebalance(mcdc):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(mcdc["bank_source"], mcdc)
    idx_end = idx_start + N_local

    distribute_work(N, mcdc)

    # Rebalance not needed if there is only one rank
    if mcdc["mpi_size"] <= 1:
        return

    # Some constants
    work_start = mcdc["mpi_work_start"]
    work_end = work_start + mcdc["mpi_work_size"]
    left = mcdc["mpi_rank"] - 1
    right = mcdc["mpi_rank"] + 1

    # Need more or less?
    more_left = idx_start < work_start
    less_left = idx_start > work_start
    more_right = idx_end > work_end
    less_right = idx_end < work_end

    # Offside?
    offside_left = idx_end <= work_start
    offside_right = idx_start >= work_end

    # MPI nearest-neighbor send/receive
    buff = np.zeros(
        mcdc["bank_source"]["particles"].shape[0], dtype=type_.particle_record
    )

    with objmode(size="int64"):
        # Create MPI-supported numpy object
        size = mcdc["bank_source"]["size"]
        bank = np.array(mcdc["bank_source"]["particles"][:size])

        # If offside, need to receive first
        if offside_left:
            # Receive from right
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
            less_right = False
        if offside_right:
            # Receive from left
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
            less_left = False

        # Send
        if more_left:
            n = work_start - idx_start
            request_left = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if more_right:
            n = idx_end - work_end
            request_right = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        # Receive
        if less_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if less_right:
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if more_left:
            request_left.Wait()
        if more_right:
            request_right.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]

    # Set source bank from buffer
    mcdc["bank_source"]["size"] = size
    for i in range(size):
        mcdc["bank_source"]["particles"][i] = buff[i]


@njit
def distribute_work(N, mcdc, precursor=False):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = math.floor(N / size)

    # Starting index (based on even distribution)
    work_start = work_size * rank

    # Count reminder
    rem = N % size

    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1
        work_start += rank
    else:
        work_start += rem

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total


# =============================================================================
# IC generator
# =============================================================================


@njit
def bank_IC(P, mcdc):
    # TODO: Consider multi-nuclide material
    material = mcdc["nuclides"][P["material_ID"]]

    # =========================================================================
    # Neutron
    # =========================================================================

    # Neutron weight
    g = P["g"]
    SigmaT = material["total"][g]
    weight = P["w"]
    flux = weight / SigmaT
    v = material["speed"][g]
    wn = flux / v

    # Neutron target weight
    Nn = mcdc["technique"]["IC_N_neutron"]
    tally_n = mcdc["technique"]["IC_neutron_density"]
    N_cycle = mcdc["setting"]["N_active"]
    wn_prime = tally_n * N_cycle / Nn

    # Sampling probability
    Pn = wn / wn_prime

    # TODO: Splitting for Pn > 1.0
    if Pn > 1.0:
        with objmode():
            print_error("Pn > 1.0.")

    # Sample particle
    if rng(P) < Pn:
        P_new = split_particle(P)
        P_new["w"] = 1.0
        P_new["t"] = 0.0
        add_particle(P_new, mcdc["technique"]["IC_bank_neutron_local"])

        # Accumulate fission
        SigmaF = material["fission"][g]
        mcdc["technique"]["IC_fission_score"] += v * SigmaF

    # =========================================================================
    # Precursor
    # =========================================================================

    # Sample precursor?
    Np = mcdc["technique"]["IC_N_precursor"]
    if Np == 0:
        return

    # Precursor weight
    J = material["J"]
    nu_d = material["nu_d"][g]
    SigmaF = material["fission"][g]
    decay = material["decay"]
    total = 0.0
    for j in range(J):
        total += nu_d[j] / decay[j]
    wp = flux * total * SigmaF / mcdc["k_eff"]

    # Material has no precursor
    if total == 0.0:
        return

    # Precursor target weight
    tally_C = mcdc["technique"]["IC_precursor_density"]
    wp_prime = tally_C * N_cycle / Np

    # Sampling probability
    Pp = wp / wp_prime

    # TODO: Splitting for Pp > 1.0
    if Pp > 1.0:
        with objmode():
            print_error("Pp > 1.0.")

    # Sample precursor
    if rng(P) < Pp:
        idx = mcdc["technique"]["IC_bank_precursor_local"]["size"]
        precursor = mcdc["technique"]["IC_bank_precursor_local"]["precursors"][idx]
        precursor["x"] = P["x"]
        precursor["y"] = P["y"]
        precursor["z"] = P["z"]
        precursor["w"] = wp_prime / wn_prime
        mcdc["technique"]["IC_bank_precursor_local"]["size"] += 1

        # Sample group
        xi = rng(P) * total
        total = 0.0
        for j in range(J):
            total += nu_d[j] / decay[j]
            if total > xi:
                break
        precursor["g"] = j

        # Set inducing neutron group
        precursor["n_g"] = g


# =============================================================================
# Population control techniques
# =============================================================================
# TODO: Make it a stand-alone function that takes (bank_init, bank_final, M).
#       The challenge is in the use of type-dependent copy_particle which is
#       required due to pure-Python behavior of taking things by reference.


@njit
def population_control(seed, mcdc):
    if mcdc["technique"]["pct"] == PCT_COMBING:
        pct_combing(seed, mcdc)
    elif mcdc["technique"]["pct"] == PCT_COMBING_WEIGHT:
        pct_combing_weight(seed, mcdc)


@njit
def pct_combing(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank
    idx_start, N_local, N = bank_scanning(bank_census, mcdc)
    idx_end = idx_start + N_local

    # Teeth distance
    td = N / M

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= td

    xi = rng_from_seed(seed)
    offset = xi * td

    # First hiting tooth
    tooth_start = math.ceil((idx_start - offset) / td)

    # Last hiting tooth
    tooth_end = math.floor((idx_end - offset) / td) + 1

    # Locally sample particles from census bank
    bank_source["size"] = 0
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx = math.floor(tooth) - idx_start
        P = copy_particle(bank_census["particles"][idx])
        # Set weight
        P["w"] *= td
        add_particle(P, bank_source)


@njit
def pct_combing_weight(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank based on weight
    w_start, w_cdf, W = bank_scanning_weight(bank_census, mcdc)
    w_end = w_cdf[-1]

    # Teeth distance
    td = W / M

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= td

    # Tooth offset
    xi = rng_from_seed(seed)
    offset = xi * td

    # First hiting tooth
    tooth_start = math.ceil((w_start - offset) / td)

    # Last hiting tooth
    tooth_end = math.floor((w_end - offset) / td) + 1

    # Locally sample particles from census bank
    bank_source["size"] = 0
    idx = 0
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx += binary_search(tooth, w_cdf[idx:])
        P = copy_particle(bank_census["particles"][idx])
        # Set weight
        P["w"] = td
        add_particle(P, bank_source)


# =============================================================================
# Particle operations
# =============================================================================


@njit
def move_particle(P, distance, mcdc):
    P["x"] += P["ux"] * distance
    P["y"] += P["uy"] * distance
    P["z"] += P["uz"] * distance
    P["t"] += distance / get_particle_speed(P, mcdc)


@njit
def shift_particle(P, shift):
    if P["ux"] > 0.0:
        P["x"] += shift
    else:
        P["x"] -= shift
    if P["uy"] > 0.0:
        P["y"] += shift
    else:
        P["y"] -= shift
    if P["uz"] > 0.0:
        P["z"] += shift
    else:
        P["z"] -= shift
    P["t"] += shift


@njit
def get_particle_cell(P, universe_ID, trans, mcdc):
    """
    Find and return particle cell ID in the given universe and translation
    """

    universe = mcdc["universes"][universe_ID]
    for cell_ID in universe["cell_IDs"]:
        cell = mcdc["cells"][cell_ID]
        if cell_check(P, cell, trans, mcdc):
            return cell["ID"]

    # Particle is not found
    with objmode():
        print("A particle is lost at (", P["x"], P["y"], P["z"], ")")
    P["alive"] = False
    return -1


@njit
def get_particle_material(P, mcdc):
    # Translation accumulator
    trans = np.zeros(3)

    # Top level cell
    cell = mcdc["cells"][P["cell_ID"]]

    # Recursively check if cell is a lattice cell, until material cell is found
    while True:
        # Lattice cell?
        if cell["lattice"]:
            # Get lattice
            lattice = mcdc["lattices"][cell["lattice_ID"]]

            # Get lattice center for translation)
            trans -= cell["lattice_center"]

            # Get universe
            mesh = lattice["mesh"]
            x, y, z = mesh_uniform_get_index(P, mesh, trans)
            universe_ID = lattice["universe_IDs"][x, y, z]

            # Update translation
            trans[0] -= mesh["x0"] + (x + 0.5) * mesh["dx"]
            trans[1] -= mesh["y0"] + (y + 0.5) * mesh["dy"]
            trans[2] -= mesh["z0"] + (z + 0.5) * mesh["dz"]

            # Get inner cell
            cell_ID = get_particle_cell(P, universe_ID, trans, mcdc)
            cell = mcdc["cells"][cell_ID]

        else:
            # Material cell found, return material_ID
            break

    return cell["material_ID"]


@njit
def get_particle_speed(P, mcdc):
    if mcdc["setting"]["mode_MG"]:
        return mcdc["materials"][P["material_ID"]]["speed"][P["g"]]
    else:
        return math.sqrt(P["E"]) * SQRT_E_TO_SPEED


@njit
def copy_particle(P):
    P_new = np.zeros(1, dtype=type_.particle_record)[0]
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["ux"] = P["ux"]
    P_new["uy"] = P["uy"]
    P_new["uz"] = P["uz"]
    P_new["g"] = P["g"]
    P_new["E"] = P["E"]
    P_new["w"] = P["w"]
    P_new["rng_seed"] = P["rng_seed"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]
    return P_new


@njit
def split_particle(P):
    P_new = copy_particle(P)
    P_new["rng_seed"] = split_seed(P["rng_seed"], SEED_SPLIT_PARTICLE)
    rng(P)
    return P_new


# =============================================================================
# Cell operations
# =============================================================================


@njit
def cell_check(P, cell, trans, mcdc):
    for i in range(cell["N_surface"]):
        surface = mcdc["surfaces"][cell["surface_IDs"][i]]
        result = surface_evaluate(P, surface, trans)
        if cell["positive_flags"][i]:
            if result < 0.0:
                return False
        else:
            if result > 0.0:
                return False
    return True


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
#   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0


@njit
def surface_evaluate(P, surface, trans):
    x = P["x"] + trans[0]
    y = P["y"] + trans[1]
    z = P["z"] + trans[2]
    t = P["t"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    # Get time indices
    idx = 0
    if surface["N_slice"] > 1:
        idx = binary_search(t, surface["t"][: surface["N_slice"] + 1])

    # Get constant
    J0 = surface["J"][idx][0]
    J1 = surface["J"][idx][1]
    J = J0 + J1 * (t - surface["t"][idx])

    result = G * x + H * y + I_ * z + J

    if surface["linear"]:
        return result

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    return (
        result + A * x * x + B * y * y + C * z * z + D * x * y + E * x * z + F * y * z
    )


@njit
def surface_bc(P, surface, trans):
    if surface["vacuum"]:
        P["alive"] = False
    elif surface["reflective"]:
        surface_reflect(P, surface, trans)


@njit
def surface_exit_evaluate(P):
    if P["surface_ID"] == 0:
        return 0
    else:
        return 1


@njit
def surface_reflect(P, surface, trans):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    nx, ny, nz = surface_normal(P, surface, trans)
    # 2.0*surface_normal_component(...)
    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    P["ux"] = ux - c * nx
    P["uy"] = uy - c * ny
    P["uz"] = uz - c * nz


@njit
def surface_shift(P, surface, trans, mcdc):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    # Get surface normal
    nx, ny, nz = surface_normal(P, surface, trans)

    # The shift
    shift_x = nx * SHIFT
    shift_y = ny * SHIFT
    shift_z = nz * SHIFT

    # Get dot product to determine shift sign
    if surface["linear"]:
        # Get time indices
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(P["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = get_particle_speed(P, mcdc)
        dot = ux * nx + uy * ny + uz * nz + J1 / v
    else:
        dot = ux * nx + uy * ny + uz * nz

    if dot > 0.0:
        P["x"] += shift_x
        P["y"] += shift_y
        P["z"] += shift_z
    else:
        P["x"] -= shift_x
        P["y"] -= shift_y
        P["z"] -= shift_z


@njit
def surface_normal(P, surface, trans):
    if surface["linear"]:
        return surface["nx"], surface["ny"], surface["nz"]

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]
    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]
    x = P["x"] + trans[0]
    y = P["y"] + trans[1]
    z = P["z"] + trans[2]

    dx = 2 * A * x + D * y + E * z + G
    dy = 2 * B * y + D * x + F * z + H
    dz = 2 * C * z + E * x + F * y + I_

    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    return dx / norm, dy / norm, dz / norm


@njit
def surface_normal_component(P, surface, trans):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    nx, ny, nz = surface_normal(P, surface, trans)
    return nx * ux + ny * uy + nz * uz


@njit
def surface_distance(P, surface, trans, mcdc):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    surface_move = False
    if surface["linear"]:
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(P["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = get_particle_speed(P, mcdc)

        t_max = surface["t"][idx + 1]
        d_max = (t_max - P["t"]) * v

        distance = -surface_evaluate(P, surface, trans) / (
            G * ux + H * uy + I_ * uz + J1 / v
        )

        # Go beyond current movement slice?
        if distance > d_max:
            distance = d_max
            surface_move = True
        elif distance < 0 and idx < surface["N_slice"] - 1:
            distance = d_max
            surface_move = True

        # Moving away from the surface
        if distance < 0.0:
            return INF, surface_move
        else:
            return distance, surface_move

    x = P["x"] + trans[0]
    y = P["y"] + trans[1]
    z = P["z"] + trans[2]

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    # Quadratic equation constants
    a = (
        A * ux * ux
        + B * uy * uy
        + C * uz * uz
        + D * ux * uy
        + E * ux * uz
        + F * uy * uz
    )
    b = (
        2 * (A * x * ux + B * y * uy + C * z * uz)
        + D * (x * uy + y * ux)
        + E * (x * uz + z * ux)
        + F * (y * uz + z * uy)
        + G * ux
        + H * uy
        + I_ * uz
    )
    c = surface_evaluate(P, surface, trans)

    determinant = b * b - 4.0 * a * c

    # Roots are complex  : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF, surface_move
    else:
        # Get the roots
        denom = 2.0 * a
        sqrt = math.sqrt(determinant)
        root_1 = (-b + sqrt) / denom
        root_2 = (-b - sqrt) / denom

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2), surface_move


# =============================================================================
# Mesh operations
# =============================================================================


@njit
def mesh_distance_search(value, direction, grid):
    if direction == 0.0:
        return INF
    idx = binary_search(value, grid)
    if direction > 0.0:
        idx += 1
    if idx == -1:
        idx += 1
    if idx == len(grid):
        idx -= 1
    dist = (grid[idx] - value) / direction
    # Moving away from mesh?
    if dist < 0.0:
        dist = INF
    return dist


@njit
def mesh_uniform_distance_search(value, direction, x0, dx):
    if direction == 0.0:
        return INF
    idx = math.floor((value - x0) / dx)
    if direction > 0.0:
        idx += 1
    ref = x0 + idx * dx
    dist = (ref - value) / direction
    return dist


@njit
def mesh_get_index(P, mesh):
    # Check if outside grid
    outside = False

    if (
        P["t"] < mesh["t"][0]
        or P["t"] > mesh["t"][-1]
        or P["x"] < mesh["x"][0]
        or P["x"] > mesh["x"][-1]
        or P["y"] < mesh["y"][0]
        or P["y"] > mesh["y"][-1]
        or P["z"] < mesh["z"][0]
        or P["z"] > mesh["z"][-1]
    ):
        outside = True
        return 0, 0, 0, 0, outside

    t = binary_search(P["t"], mesh["t"])
    x = binary_search(P["x"], mesh["x"])
    y = binary_search(P["y"], mesh["y"])
    z = binary_search(P["z"], mesh["z"])
    return t, x, y, z, outside


@njit
def mesh_get_angular_index(P, mesh):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    P_mu = uz
    P_azi = math.acos(ux / math.sqrt(ux * ux + uy * uy))
    if uy < 0.0:
        P_azi *= -1

    mu = binary_search(P_mu, mesh["mu"])
    azi = binary_search(P_azi, mesh["azi"])
    return mu, azi


@njit
def mesh_get_energy_index(P, mesh, mcdc):
    # Check if outside grid
    outside = False

    if mcdc["setting"]["mode_MG"]:
        return binary_search(P["g"], mesh["g"]), outside
    else:
        E = P["E"]
        if E < mesh["g"][0] or E > mesh["g"][-1]:
            outside = True
            return 0, outside
        return binary_search(P["E"], mesh["g"]), outside


@njit
def mesh_uniform_get_index(P, mesh, trans):
    Px = P["x"] + trans[0]
    Py = P["y"] + trans[1]
    Pz = P["z"] + trans[2]
    x = math.floor((Px - mesh["x0"]) / mesh["dx"])
    y = math.floor((Py - mesh["y0"]) / mesh["dy"])
    z = math.floor((Pz - mesh["z0"]) / mesh["dz"])
    return x, y, z


# =============================================================================
# Tally operations
# =============================================================================


@njit
def score_tracklength(P, distance, mcdc):
    tally = mcdc["tally"]
    material = mcdc["materials"][P["material_ID"]]

    # Get indices
    s = P["sensitivity_ID"]
    t, x, y, z, outside = mesh_get_index(P, tally["mesh"])
    mu, azi = mesh_get_angular_index(P, tally["mesh"])
    g, outside_energy = mesh_get_energy_index(P, tally["mesh"], mcdc)

    # Outside grid?
    if outside or outside_energy:
        return

    # Score
    flux = distance * P["w"]
    if tally["flux"]:
        score_flux(s, g, t, x, y, z, mu, azi, flux, tally["score"]["flux"])
    if tally["density"]:
        flux /= material["speed"][g]
        score_flux(s, g, t, x, y, z, mu, azi, flux, tally["score"]["density"])
    if tally["fission"]:
        flux *= material["fission"][g]
        score_flux(s, g, t, x, y, z, mu, azi, flux, tally["score"]["fission"])
    if tally["total"]:
        flux *= material["total"][g]
        score_flux(s, g, t, x, y, z, mu, azi, flux, tally["score"]["total"])
    if tally["current"]:
        score_current(s, g, t, x, y, z, flux, P, tally["score"]["current"])
    if tally["eddington"]:
        score_eddington(s, g, t, x, y, z, flux, P, tally["score"]["eddington"])


@njit
def score_exit(P, x, mcdc):
    tally = mcdc["tally"]
    material = mcdc["materials"][P["material_ID"]]

    s = P["sensitivity_ID"]
    mu, azi = mesh_get_angular_index(P, tally["mesh"])
    g, outside_energy = mesh_get_energy_index(P, tally["mesh"], mcdc)

    # Outside grid?
    if outside_energy:
        return

    # Score
    flux = P["w"] / abs(P["ux"])
    score_flux(s, g, 0, x, 0, 0, mu, azi, flux, tally["score"]["exit"])


@njit
def score_flux(s, g, t, x, y, z, mu, azi, flux, score):
    score["bin"][s, g, t, x, y, z, mu, azi] += flux


@njit
def score_current(s, g, t, x, y, z, flux, P, score):
    score["bin"][s, g, t, x, y, z, 0] += flux * P["ux"]
    score["bin"][s, g, t, x, y, z, 1] += flux * P["uy"]
    score["bin"][s, g, t, x, y, z, 2] += flux * P["uz"]


@njit
def score_eddington(s, g, t, x, y, z, flux, P, score):
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    score["bin"][s, g, t, x, y, z, 0] += flux * ux * ux
    score["bin"][s, g, t, x, y, z, 1] += flux * ux * uy
    score["bin"][s, g, t, x, y, z, 2] += flux * ux * uz
    score["bin"][s, g, t, x, y, z, 3] += flux * uy * uy
    score["bin"][s, g, t, x, y, z, 4] += flux * uy * uz
    score["bin"][s, g, t, x, y, z, 5] += flux * uz * uz


@njit
def score_reduce_bin(score, mcdc):
    # Normalize
    score["bin"][:] /= mcdc["setting"]["N_particle"]

    # MPI Reduce
    buff = np.zeros_like(score["bin"])
    with objmode():
        MPI.COMM_WORLD.Reduce(np.array(score["bin"]), buff, MPI.SUM, 0)
    score["bin"][:] = buff


@njit
def score_closeout_history(score):
    # Accumulate score and square of score into mean and sdev
    score["mean"][:] += score["bin"]
    score["sdev"][:] += np.square(score["bin"])

    # Reset bin
    score["bin"].fill(0.0)


@njit
def score_closeout(score, mcdc):
    N_history = mcdc["setting"]["N_particle"]

    if mcdc["setting"]["N_batch"] > 1:
        N_history = mcdc["setting"]["N_batch"]

    elif mcdc["setting"]["mode_eigenvalue"]:
        N_history = mcdc["setting"]["N_active"]

    else:
        # MPI Reduce
        buff = np.zeros_like(score["mean"])
        buff_sq = np.zeros_like(score["sdev"])
        with objmode():
            MPI.COMM_WORLD.Reduce(np.array(score["mean"]), buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(np.array(score["sdev"]), buff_sq, MPI.SUM, 0)
        score["mean"][:] = buff
        score["sdev"][:] = buff_sq

    # Store results
    score["mean"][:] = score["mean"] / N_history
    score["sdev"][:] = np.sqrt(
        (score["sdev"] / N_history - np.square(score["mean"])) / (N_history - 1)
    )


@njit
def tally_reduce_bin(mcdc):
    tally = mcdc["tally"]

    for name in literal_unroll(score_list):
        if tally[name]:
            score_reduce_bin(tally["score"][name], mcdc)


@njit
def tally_closeout_history(mcdc):
    tally = mcdc["tally"]

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout_history(tally["score"][name])


@njit
def tally_closeout(mcdc):
    tally = mcdc["tally"]

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout(tally["score"][name], mcdc)


# =============================================================================
# Eigenvalue tally operations
# =============================================================================


@njit
def eigenvalue_tally(P, distance, mcdc):
    tally = mcdc["tally"]

    # TODO: Consider multi-nuclide material
    material = mcdc["nuclides"][P["material_ID"]]

    # Parameters
    flux = distance * P["w"]
    g = P["g"]
    nu = material["nu_f"][g]
    SigmaT = material["total"][g]
    SigmaF = material["fission"][g]
    nuSigmaF = nu * SigmaF

    # Fission production (needed even during inactive cycle)
    mcdc["eigenvalue_tally_nuSigmaF"] += flux * nuSigmaF

    if mcdc["cycle_active"]:
        # Neutron density
        v = get_particle_speed(P, mcdc)
        n_density = flux / v
        mcdc["eigenvalue_tally_n"] += n_density
        # Maximum neutron density
        if mcdc["n_max"] < n_density:
            mcdc["n_max"] = n_density

        # Precursor density
        J = material["J"]
        nu_d = material["nu_d"][g]
        decay = material["decay"]
        total = 0.0
        for j in range(J):
            total += nu_d[j] / decay[j]
        C_density = flux * total * SigmaF / mcdc["k_eff"]
        mcdc["eigenvalue_tally_C"] += C_density
        # Maximum precursor density
        if mcdc["C_max"] < C_density:
            mcdc["C_max"] = C_density


@njit
def eigenvalue_tally_closeout_history(mcdc):
    N_particle = mcdc["setting"]["N_particle"]

    idx_cycle = mcdc["idx_cycle"]

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_n = np.zeros(1, np.float64)
    buff_nmax = np.zeros(1, np.float64)
    buff_C = np.zeros(1, np.float64)
    buff_Cmax = np.zeros(1, np.float64)
    buff_IC_fission = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(
            np.array([mcdc["eigenvalue_tally_nuSigmaF"]]), buff_nuSigmaF, MPI.SUM
        )
        if mcdc["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array([mcdc["eigenvalue_tally_n"]]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["n_max"]]), buff_nmax, MPI.MAX)
            MPI.COMM_WORLD.Allreduce(
                np.array([mcdc["eigenvalue_tally_C"]]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["C_max"]]), buff_Cmax, MPI.MAX)
            if mcdc["technique"]["IC_generator"]:
                MPI.COMM_WORLD.Allreduce(
                    np.array([mcdc["technique"]["IC_fission_score"]]),
                    buff_IC_fission,
                    MPI.SUM,
                )

    # Update and store k_eff
    mcdc["k_eff"] = buff_nuSigmaF[0] / N_particle
    mcdc["k_cycle"][idx_cycle] = mcdc["k_eff"]

    # Normalize other eigenvalue/global tallies
    tally_n = buff_n[0] / N_particle
    tally_C = buff_C[0] / N_particle
    tally_IC_fission = buff_IC_fission[0]

    # Maximum densities
    mcdc["n_max"] = buff_nmax[0]
    mcdc["C_max"] = buff_Cmax[0]

    # Accumulate running average
    if mcdc["cycle_active"]:
        mcdc["k_avg"] += mcdc["k_eff"]
        mcdc["k_sdv"] += mcdc["k_eff"] * mcdc["k_eff"]
        mcdc["n_avg"] += tally_n
        mcdc["n_sdv"] += tally_n * tally_n
        mcdc["C_avg"] += tally_C
        mcdc["C_sdv"] += tally_C * tally_C

        N = 1 + mcdc["idx_cycle"] - mcdc["setting"]["N_inactive"]
        mcdc["k_avg_running"] = mcdc["k_avg"] / N
        if N == 1:
            mcdc["k_sdv_running"] = 0.0
        else:
            mcdc["k_sdv_running"] = math.sqrt(
                (mcdc["k_sdv"] / N - mcdc["k_avg_running"] ** 2) / (N - 1)
            )

        if mcdc["technique"]["IC_generator"]:
            mcdc["technique"]["IC_fission"] += tally_IC_fission

    # Reset accumulators
    mcdc["eigenvalue_tally_nuSigmaF"] = 0.0
    mcdc["eigenvalue_tally_n"] = 0.0
    mcdc["eigenvalue_tally_C"] = 0.0
    mcdc["technique"]["IC_fission_score"] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if mcdc["setting"]["gyration_radius"]:
        # Center of mass
        N_local = mcdc["bank_census"]["size"]
        total_local = np.zeros(4, np.float64)  # [x,y,z,W]
        total = np.zeros(4, np.float64)
        for i in range(N_local):
            P = mcdc["bank_census"]["particles"][i]
            total_local[0] += P["x"] * P["w"]
            total_local[1] += P["y"] * P["w"]
            total_local[2] += P["z"] * P["w"]
            total_local[3] += P["w"]
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W = total[3]
        com_x = total[0] / W
        com_y = total[1] / W
        com_z = total[2] / W

        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms = np.zeros(1, np.float64)
        gr_type = mcdc["setting"]["gyration_radius_type"]
        if gr_type == GYRATION_RADIUS_ALL:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += (
                    (P["x"] - com_x) ** 2
                    + (P["y"] - com_y) ** 2
                    + (P["z"] - com_z) ** 2
                ) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["z"] - com_z) ** 2) * P["w"]

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0] / W)

        # Gyration radius
        mcdc["gyration_radius"][idx_cycle] = rms


@njit
def eigenvalue_tally_closeout(mcdc):
    N = mcdc["setting"]["N_active"]
    mcdc["n_avg"] /= N
    mcdc["C_avg"] /= N
    if N > 1:
        mcdc["n_sdv"] = math.sqrt((mcdc["n_sdv"] / N - mcdc["n_avg"] ** 2) / (N - 1))
        mcdc["C_sdv"] = math.sqrt((mcdc["C_sdv"] / N - mcdc["C_avg"] ** 2) / (N - 1))
    else:
        mcdc["n_sdv"] = 0.0
        mcdc["C_sdv"] = 0.0


# =============================================================================
# Move to event
# =============================================================================


@njit
def move_to_event(P, mcdc):
    # =========================================================================
    # Get distances to events
    # =========================================================================

    # Distance to nearest geometry boundary (surface or lattice)
    # Also set particle material and speed
    d_boundary, event = distance_to_boundary(P, mcdc)

    # Distance to tally mesh
    d_mesh = INF
    if mcdc["cycle_active"]:
        d_mesh = distance_to_mesh(P, mcdc["tally"]["mesh"], mcdc)

    if mcdc["technique"]["iQMC"]:
        d_iqmc_mesh = distance_to_mesh(P, mcdc["technique"]["iqmc"]["mesh"], mcdc)
        if d_iqmc_mesh < d_mesh:
            d_mesh = d_iqmc_mesh

    # Distance to time boundary
    speed = get_particle_speed(P, mcdc)
    d_time_boundary = speed * (mcdc["setting"]["time_boundary"] - P["t"])

    # Distance to census time
    idx = mcdc["idx_census"]
    d_time_census = speed * (mcdc["setting"]["census_time"][idx] - P["t"])

    # Distance to collision
    if mcdc["technique"]["iQMC"]:
        d_collision = INF
    else:
        d_collision = distance_to_collision(P, mcdc)

    # =========================================================================
    # Determine event
    #   Priority (in case of coincident events):
    #     boundary > time_boundary > mesh > collision
    # =========================================================================

    # Find the minimum
    distance = min(d_boundary, d_time_boundary, d_time_census, d_mesh, d_collision)
    # Remove the boundary event if it is not the nearest
    if d_boundary > distance * PREC:
        event = 0

    # Add each event if it is within PREC of the nearest event
    if d_time_boundary <= distance * PREC:
        event += EVENT_TIME_BOUNDARY
    if d_time_census <= distance * PREC:
        event += EVENT_CENSUS
    if d_mesh <= distance * PREC:
        event += EVENT_MESH
    if d_collision == distance:
        event = EVENT_COLLISION

    # Assign event
    P["event"] = event

    # =========================================================================
    # Move particle
    # =========================================================================

    # score iQMC tallies
    if mcdc["technique"]["iQMC"]:
        if mcdc["setting"]["track_particle"]:
            track_particle(P, mcdc)
        iqmc_score_tallies(P, distance, mcdc)
        iqmc_continuous_weight_reduction(P, distance, mcdc)
        if np.abs(P["w"]) <= mcdc["technique"]["iqmc"]["w_min"]:
            P["alive"] = False

    # Score tracklength tallies
    if mcdc["tally"]["tracklength"] and mcdc["cycle_active"]:
        score_tracklength(P, distance, mcdc)
    if mcdc["setting"]["mode_eigenvalue"]:
        eigenvalue_tally(P, distance, mcdc)

    # Move particle
    move_particle(P, distance, mcdc)


@njit
def distance_to_collision(P, mcdc):
    # Get total cross-section
    material = mcdc["materials"][P["material_ID"]]
    SigmaT = get_MacroXS(XS_TOTAL, material, P, mcdc)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng(P)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def distance_to_boundary(P, mcdc):
    """
    Find the nearest geometry boundary, which could be lattice or surface, and
    return the event type (EVENT_SURFACE or EVENT_LATTICE) and the distance

    We recursively check from the top level cell. If surface and lattice are
    coincident, EVENT_SURFACE is prioritized.
    """

    distance = INF
    event = 0

    # Translation accumulator
    trans = np.zeros(3)

    # Top level cell
    cell = mcdc["cells"][P["cell_ID"]]

    # Recursively check if cell is a lattice cell, until material cell is found
    while True:
        # Distance to nearest surface
        d_surface, surface_ID, surface_move = distance_to_nearest_surface(
            P, cell, trans, mcdc
        )

        # Check if smaller
        if d_surface * PREC < distance:
            distance = d_surface
            event = EVENT_SURFACE
            P["surface_ID"] = surface_ID
            P["translation"][:] = trans

            if surface_move:
                event = EVENT_SURFACE_MOVE

        # Lattice cell?
        if cell["lattice"]:
            # Get lattice
            lattice = mcdc["lattices"][cell["lattice_ID"]]

            # Get lattice center for translation)
            trans -= cell["lattice_center"]

            # Distance to lattice
            d_lattice = distance_to_lattice(P, lattice, trans)

            # Check if smaller
            if d_lattice * PREC < distance:
                distance = d_lattice
                event = EVENT_LATTICE
                P["surface_ID"] = -1

            # Get universe
            mesh = lattice["mesh"]
            x, y, z = mesh_uniform_get_index(P, mesh, trans)
            universe_ID = lattice["universe_IDs"][x, y, z]

            # Update translation
            trans[0] -= mesh["x0"] + (x + 0.5) * mesh["dx"]
            trans[1] -= mesh["y0"] + (y + 0.5) * mesh["dy"]
            trans[2] -= mesh["z0"] + (z + 0.5) * mesh["dz"]

            # Get inner cell
            cell_ID = get_particle_cell(P, universe_ID, trans, mcdc)
            cell = mcdc["cells"][cell_ID]

        else:
            # Material cell found, set material_ID
            P["material_ID"] = cell["material_ID"]
            break

    return distance, event


@njit
def distance_to_nearest_surface(P, cell, trans, mcdc):
    distance = INF
    surface_ID = -1
    surface_move = False

    for i in range(cell["N_surface"]):
        surface = mcdc["surfaces"][cell["surface_IDs"][i]]
        d, sm = surface_distance(P, surface, trans, mcdc)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]
            surface_move = sm
    return distance, surface_ID, surface_move


@njit
def distance_to_lattice(P, lattice, trans):
    mesh = lattice["mesh"]

    x = P["x"] + trans[0]
    y = P["y"] + trans[1]
    z = P["z"] + trans[2]
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    d = INF
    d = min(d, mesh_uniform_distance_search(x, ux, mesh["x0"], mesh["dx"]))
    d = min(d, mesh_uniform_distance_search(y, uy, mesh["y0"], mesh["dy"]))
    d = min(d, mesh_uniform_distance_search(z, uz, mesh["z0"], mesh["dz"]))
    return d


@njit
def distance_to_mesh(P, mesh, mcdc):
    x = P["x"]
    y = P["y"]
    z = P["z"]
    t = P["t"]
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    v = get_particle_speed(P, mcdc)

    d = INF
    d = min(d, mesh_distance_search(x, ux, mesh["x"]))
    d = min(d, mesh_distance_search(y, uy, mesh["y"]))
    d = min(d, mesh_distance_search(z, uz, mesh["z"]))
    d = min(d, mesh_distance_search(t, 1.0 / v, mesh["t"]))
    return d


# =============================================================================
# Surface crossing
# =============================================================================


@njit
def surface_crossing(P, mcdc):
    trans = P["translation"]

    # Implement BC
    surface = mcdc["surfaces"][P["surface_ID"]]
    surface_bc(P, surface, trans)

    # Small shift to ensure crossing
    surface_shift(P, surface, trans, mcdc)

    # Record old material for sensitivity quantification
    material_ID_old = P["material_ID"]

    # Tally particle exit
    if mcdc["tally"]["exit"] and not P["alive"]:
        # Reflectance if P["surface_ID"] == 0, else transmittance
        exit_idx = surface_exit_evaluate(P)
        # Score on tally
        score_exit(P, exit_idx, mcdc)

    # Check new cell?
    if P["alive"] and not surface["reflective"]:
        cell = mcdc["cells"][P["cell_ID"]]
        if not cell_check(P, cell, trans, mcdc):
            trans = np.zeros(3)
            P["cell_ID"] = get_particle_cell(P, 0, trans, mcdc)

    # Sensitivity quantification for surface?
    if surface["sensitivity"] and (
        P["sensitivity_ID"] == 0
        or mcdc["technique"]["dsm_order"] == 2
        and P["sensitivity_ID"] <= mcdc["setting"]["N_sensitivity"]
    ):
        material_ID_new = get_particle_material(P, mcdc)
        if material_ID_old != material_ID_new:
            # Sample derivative source particles
            sensitivity_surface(P, surface, material_ID_old, material_ID_new, mcdc)


# =============================================================================
# Collision
# =============================================================================


@njit
def collision(P, mcdc):
    # Get the reaction cross-sections
    material = mcdc["materials"][P["material_ID"]]
    g = P["g"]
    SigmaT = get_MacroXS(XS_TOTAL, material, P, mcdc)
    SigmaS = get_MacroXS(XS_SCATTER, material, P, mcdc)
    SigmaC = get_MacroXS(XS_CAPTURE, material, P, mcdc)
    SigmaF = get_MacroXS(XS_FISSION, material, P, mcdc)

    # Implicit capture
    if mcdc["technique"]["implicit_capture"]:
        P["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample collision type
    xi = rng(P) * SigmaT
    tot = SigmaS
    if tot > xi:
        event = EVENT_SCATTERING
    else:
        tot += SigmaF
        if tot > xi:
            event = EVENT_FISSION
        else:
            event = EVENT_CAPTURE
    P["event"] = event

    # =========================================================================
    # Implement minor events
    # =========================================================================

    if event & EVENT_CAPTURE:
        P["alive"] = False


# =============================================================================
# Capture
# =============================================================================


@njit
def capture(P, mcdc):
    # Kill the current particle
    P["alive"] = False


# =============================================================================
# Scattering
# =============================================================================


@njit
def scattering(P, mcdc):
    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Get number of secondaries
    material = mcdc["materials"][P["material_ID"]]
    g = P["g"]
    if mcdc["setting"]["mode_MG"]:
        nu_s = material["nu_s"][g]
        N = int(math.floor(weight_eff * nu_s + rng(P)))
    else:
        N = 1

    for n in range(N):
        # Create new particle
        P_new = split_particle(P)

        # Set weight
        P_new["w"] = weight_new

        # Sample scattering phase space
        sample_phasespace_scattering(P, material, P_new, mcdc)

        # Bank
        add_particle(P_new, mcdc["bank_active"])


@njit
def sample_phasespace_scattering(P, material, P_new, mcdc):
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]

    if mcdc["setting"]["mode_MG"]:
        scattering_MG(P, material, P_new)
    else:
        scattering_CE(P, material, P_new, mcdc)


@njit
def sample_phasespace_scattering_nuclide(P, nuclide, P_new, mcdc):
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]

    scattering_MG(P, nuclide, P_new)


@njit
def scattering_MG(P, material, P_new):
    # Sample scattering angle
    mu0 = 2.0 * rng(P_new) - 1.0

    # Scatter direction
    azi = 2.0 * PI * rng(P_new)
    P_new["ux"], P_new["uy"], P_new["uz"] = scatter_direction(
        P["ux"], P["uy"], P["uz"], mu0, azi
    )

    # Get outgoing spectrum
    g = P["g"]
    G = material["G"]
    chi_s = material["chi_s"][g]

    # Sample outgoing energy
    xi = rng(P_new)
    tot = 0.0
    for g_out in range(G):
        tot += chi_s[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out


@njit
def scattering_CE(P, material, P_new, mcdc):
    """
    Scatter with sampled scattering angle mu0, with nucleus mass A
    Scattering is treated in Center of mass (COM) frame
    Current model:
      - Free gas scattering
      - Constant thermal cross section
      - Isotropic in COM
    """
    # Sample nuclide
    nuclide = sample_nuclide(material, P, XS_SCATTER, mcdc)
    xi = rng(P) * get_MacroXS(XS_SCATTER, material, P, mcdc)
    tot = 0.0
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]
        N = material["nuclide_densities"][i]
        tot += N * get_microXS(XS_SCATTER, nuclide, P["E"])
        if tot > xi:
            break

    # Sample nucleus thermal speed
    A = nuclide["A"]
    if P["E"] > E_THERMAL_THRESHOLD:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_speed(A, P, mcdc)

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    P_speed = get_particle_speed(P, mcdc)

    # Neutron velocity - LAB
    vx = P_speed * P["ux"]
    vy = P_speed * P["uy"]
    vz = P_speed * P["uz"]

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Neutron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Neutron speed - COM
    P_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Neutron initial direction - COM
    ux = vx / P_speed
    uy = vy / P_speed
    uz = vz / P_speed

    # Scatter the direction in COM
    mu0 = 2.0 * rng(P) - 1.0
    azi = 2.0 * PI * rng(P)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Neutron final velocity - COM
    vx = P_speed * ux_new
    vy = P_speed * uy_new
    vz = P_speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx = vx + COM_x
    vy = vy + COM_y
    vz = vz + COM_z

    # Final energy - LAB
    P_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    P_new["E"] = 5.2270376e-13 * P_speed * P_speed
    # constant: 0.5 / (1.60217662e-19 J/eV) * (1.674927471e-27 kg) / (10000 cm^2/m^2)

    # Final direction - LAB
    P_new["ux"] = vx / P_speed
    P_new["uy"] = vy / P_speed
    P_new["uz"] = vz / P_speed


@njit
def sample_nucleus_speed(A, P, mcdc):
    # Particle speed
    P_speed = get_particle_speed(P, mcdc)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * P_speed
    while True:
        if rng(P) < 2.0 / (2.0 + PI_SQRT * y):
            x = math.sqrt(-math.log(rng(P) * rng(P)))
        else:
            cos_val = math.cos(PI_HALF * rng(P))
            x = math.sqrt(-math.log(rng(P)) - math.log(rng(P)) * cos_val * cos_val)
        V_tilda = x / beta
        mu_tilda = 2.0 * rng(P) - 1.0

        # Accept candidate V_tilda and mu_tilda?
        if rng(P) > math.sqrt(
            P_speed * P_speed + V_tilda * V_tilda - 2.0 * P_speed * V_tilda * mu_tilda
        ) / (P_speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * rng(P)
    ux, uy, uz = scatter_direction(P["ux"], P["uy"], P["uz"], mu_tilda, azi)
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


@njit
def scatter_direction(ux, uy, uz, mu0, azi):
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu0**2) ** 0.5

    if uz != 1.0:
        B = (1.0 - uz**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uz * cos_azi - uy * sin_azi) * C
        uy_new = uy * mu0 + (uy * uz * cos_azi + ux * sin_azi) * C
        uz_new = uz * mu0 - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uy * cos_azi - uz * sin_azi) * C
        uz_new = uz * mu0 + (uz * uy * cos_azi + ux * sin_azi) * C
        uy_new = uy * mu0 - cos_azi * Ac * B

    return ux_new, uy_new, uz_new


# =============================================================================
# Fission
# =============================================================================


@njit
def fission(P, mcdc):
    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Sample nuclide if CE
    material = mcdc["materials"][P["material_ID"]]
    nuclide = mcdc["nuclides"][0]  # Default nuclide, will be resampled for CE

    # Get number of secondaries
    if mcdc["setting"]["mode_MG"]:
        g = P["g"]
        nu = material["nu_f"][g]
    else:
        nuclide = sample_nuclide(material, P, XS_FISSION, mcdc)
        E = P["E"]
        nu = get_nu(NU_TOTAL, nuclide, E)
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + rng(P)))

    for n in range(N):
        # Create new particle
        P_new = split_particle(P)

        # Set weight
        P_new["w"] = weight_new

        # Sample fission neutron phase space
        if mcdc["setting"]["mode_MG"]:
            sample_phasespace_fission(P, material, P_new, mcdc)
        else:
            sample_phasespace_fission_nuclide(P, nuclide, P_new, mcdc)

        # Skip if it's beyond time boundary
        if P_new["t"] > mcdc["setting"]["time_boundary"]:
            continue

        # Bank
        if mcdc["setting"]["mode_eigenvalue"]:
            add_particle(P_new, mcdc["bank_census"])
        else:
            add_particle(P_new, mcdc["bank_active"])


@njit
def sample_phasespace_fission(P, material, P_new, mcdc):
    # Get constants
    G = material["G"]
    J = material["J"]
    g = P["g"]
    nu = material["nu_f"][g]
    nu_p = material["nu_p"][g]
    if J > 0:
        nu_d = material["nu_d"][g]

    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]

    # Sample isotropic direction
    P_new["ux"], P_new["uy"], P_new["uz"] = sample_isotropic_direction(P_new)

    # Prompt or delayed?
    xi = rng(P_new) * nu
    tot = nu_p
    if xi < tot:
        prompt = True
        spectrum = material["chi_p"][g]
    else:
        prompt = False

        # Determine delayed group and nuclide-dependent decay constant and spectrum
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                # Delayed group determined, now determine nuclide
                N_nuclide = material["N_nuclide"]
                if N_nuclide == 1:
                    nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
                    spectrum = nuclide["chi_d"][j]
                    decay = nuclide["decay"][j]
                    break
                SigmaF = get_MacroXS(XS_FISSION, material, P, mcdc)
                xi = rng(P_new) * nu_d[j] * SigmaF
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
                break

    # Sample outgoing energy
    xi = rng(P_new)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out

    # Sample emission time
    if not prompt:
        xi = rng(P_new)
        P_new["t"] -= math.log(xi) / decay


@njit
def sample_phasespace_fission_nuclide(P, nuclide, P_new, mcdc):
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]

    # Sample isotropic direction
    P_new["ux"], P_new["uy"], P_new["uz"] = sample_isotropic_direction(P_new)

    if mcdc["setting"]["mode_MG"]:
        fission_MG(P, nuclide, P_new)
    else:
        fission_CE(P, nuclide, P_new)


@njit
def fission_MG(P, nuclide, P_new):
    # Get constants
    G = nuclide["G"]
    J = nuclide["J"]
    g = P["g"]
    nu = nuclide["nu_f"][g]
    nu_p = nuclide["nu_p"][g]
    if J > 0:
        nu_d = nuclide["nu_d"][g]

    # Prompt or delayed?
    xi = rng(P_new) * nu
    tot = nu_p
    if xi < tot:
        prompt = True
        spectrum = nuclide["chi_p"][g]
    else:
        prompt = False

        # Determine delayed group
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                spectrum = nuclide["chi_d"][j]
                decay = nuclide["decay"][j]
                break

    # Sample outgoing energy
    xi = rng(P_new)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out

    # Sample emission time
    if not prompt:
        xi = rng(P_new)
        P_new["t"] -= math.log(xi) / decay


@njit
def fission_CE(P, nuclide, P_new):
    # Get constants
    E = P["E"]
    J = 6
    nu = get_nu(NU_TOTAL, nuclide, E)
    nu_p = get_nu(NU_PROMPT, nuclide, E)
    nu_d = np.zeros(J)
    for j in range(J):
        nu_d[j] = get_nu(NU_DELAYED, nuclide, E, j)

    # Delayed?
    prompt = True
    delayed_group = -1
    xi = rng(P_new) * nu
    tot = nu_p
    if xi > tot:
        prompt = False

        # Determine delayed group
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                delayed_group = j
                break

    # Sample outgoing energy
    if prompt:
        E_chi = nuclide["E_chi_p"]
        NE_chi = nuclide["NE_chi_p"]
        chi = nuclide["ce_chi_p"]
        P_new["E"] = sample_Eout(
            P_new, nuclide["E_chi_p"], nuclide["NE_chi_p"], nuclide["ce_chi_p"]
        )
    else:
        if delayed_group == 0:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d1"], nuclide["NE_chi_d1"], nuclide["ce_chi_d1"]
            )
        elif delayed_group == 1:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d2"], nuclide["NE_chi_d2"], nuclide["ce_chi_d2"]
            )
        elif delayed_group == 2:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d3"], nuclide["NE_chi_d3"], nuclide["ce_chi_d3"]
            )
        elif delayed_group == 3:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d4"], nuclide["NE_chi_d4"], nuclide["ce_chi_d4"]
            )
        elif delayed_group == 4:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d5"], nuclide["NE_chi_d5"], nuclide["ce_chi_d5"]
            )
        else:
            P_new["E"] = sample_Eout(
                P_new, nuclide["E_chi_d6"], nuclide["NE_chi_d6"], nuclide["ce_chi_d6"]
            )

    # Sample emission time
    if not prompt:
        xi = rng(P_new)
        P_new["t"] -= math.log(xi) / nuclide["ce_decay"][delayed_group]


# =============================================================================
# Branchless collision
# =============================================================================


@njit
def branchless_collision(P, mcdc):
    # Data
    # TODO: Consider multi-nuclide material
    material = mcdc["nuclides"][P["material_ID"]]
    w = P["w"]
    g = P["g"]
    SigmaT = material["total"][g]
    SigmaS = material["scatter"][g]
    SigmaF = material["fission"][g]
    nu_s = material["nu_s"][g]
    nu_p = material["nu_p"][g] / mcdc["k_eff"]
    nu_d = material["nu_d"][g] / mcdc["k_eff"]
    J = material["J"]
    G = material["G"]

    # Total nu fission
    nu = material["nu_f"][g]

    # Set weight
    n_scatter = nu_s * SigmaS
    n_fission = nu * SigmaF
    n_total = n_fission + n_scatter
    P["w"] *= n_total / SigmaT

    # Set spectrum and decay rate
    fission = True
    prompt = True
    if rng(P) < n_scatter / n_total:
        fission = False
        spectrum = material["chi_s"][g]
    else:
        xi = rng(P) * nu
        tot = nu_p
        if xi < tot:
            spectrum = material["chi_p"][g]
        else:
            prompt = False
            for j in range(J):
                tot += nu_d[j]
                if xi < tot:
                    spectrum = material["chi_d"][j]
                    decay = material["decay"][j]
                    break

    # Set time
    if not prompt:
        xi = rng(P)
        P["t"] -= math.log(xi) / decay

        # Kill if it's beyond time boundary
        if P["t"] > mcdc["setting"]["time_boundary"]:
            P["alive"] = False
            return

    # Set energy
    xi = rng(P)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            P["g"] = g_out
            break

    # Set direction (TODO: anisotropic scattering)
    P["ux"], P["uy"], P["uz"] = sample_isotropic_direction(P)


# =============================================================================
# Weight widow
# =============================================================================


@njit
def weight_window(P, mcdc):
    # Get indices
    t, x, y, z, outside = mesh_get_index(P, mcdc["technique"]["ww_mesh"])

    # Target weight
    w_target = mcdc["technique"]["ww"][t, x, y, z]

    # Population control factor
    w_target *= mcdc["technique"]["pc_factor"]

    # Surviving probability
    p = P["w"] / w_target

    # Window width
    width = mcdc["technique"]["ww_width"]

    # If above target
    if p > width:
        # Set target weight
        P["w"] = w_target

        # Splitting (keep the original particle)
        n_split = math.floor(p)
        for i in range(n_split - 1):
            add_particle(split_particle(P), mcdc["bank_active"])

        # Russian roulette
        p -= n_split
        xi = rng(P)
        if xi <= p:
            add_particle(split_particle(P), mcdc["bank_active"])

    # Below target
    elif p < 1.0 / width:
        # Russian roulette
        xi = rng(P)
        if xi > p:
            P["alive"] = False
        else:
            P["w"] = w_target


# ==============================================================================
# Quasi Monte Carlo
# ==============================================================================


@njit
def iqmc_continuous_weight_reduction(P, distance, mcdc):
    """
    Continuous weight reduction technique based on particle track-length, for
    use with iQMC.

    Parameters
    ----------
    w : float64
        particle weight
    distance : float64
        track length
    SigmaT : float64
        total cross section

    Returns
    -------
    float64
        New particle weight
    """
    material = mcdc["materials"][P["material_ID"]]
    SigmaT = material["total"][:]
    w = P["iqmc"]["w"]
    P["iqmc"]["w"] = w * np.exp(-distance * SigmaT)
    P["w"] = P["iqmc"]["w"].sum()


@njit
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


@njit
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


@njit
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


@njit
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
        P_new = np.zeros(1, dtype=type_.particle_record)[0]
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
        add_particle(P_new, mcdc["bank_source"])


@njit
def iqmc_res(flux_new, flux_old):
    """

    Calculate residual between scalar flux iterations.

    Parameters
    ----------
    flux_new : TYPE
        Current scalar flux iteration.
    flux_old : TYPE
        previous scalar flux iteration.

    Returns
    -------
    float64
        L2 Norm of arrays.

    """
    size = flux_old.size
    flux_new = np.linalg.norm(flux_new.reshape((size,)), ord=2)
    flux_old = np.linalg.norm(flux_old.reshape((size,)), ord=2)
    return (flux_new - flux_old) / flux_old


@njit
def iqmc_score_tallies(P, distance, mcdc):
    """

    Tally the scalar flux and linear source tilt.

    Parameters
    ----------
    P : particle
    distance : float64
        tracklength.
    mcdc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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

    if score_list["tilt-xy"]:
        tilt = iqmc_bilinear_tilt(
            P["ux"],
            P["x"],
            dx,
            x_mid,
            P["uy"],
            P["y"],
            dy,
            y_mid,
            dt,
            dz,
            w,
            distance,
            SigmaT,
        )
        score_bin["tilt-xy"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)

    if score_list["tilt-xz"]:
        tilt = iqmc_bilinear_tilt(
            P["ux"],
            P["x"],
            dx,
            x_mid,
            P["uz"],
            P["z"],
            dz,
            z_mid,
            dt,
            dy,
            w,
            distance,
            SigmaT,
        )
        score_bin["tilt-xz"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)

    if score_list["tilt-yz"]:
        tilt = iqmc_bilinear_tilt(
            P["uy"],
            P["y"],
            dy,
            y_mid,
            P["uz"],
            P["z"],
            dz,
            z_mid,
            dt,
            dx,
            w,
            distance,
            SigmaT,
        )
        score_bin["tilt-yz"][:, t, x, y, z] += iqmc_effective_source(tilt, mat_id, mcdc)


@njit
def iqmc_cell_volume(x, y, z, mesh):
    """
    Calculate the volume of the current spatial cell.

    Parameters
    ----------
    x : int64
        Current x-position index.
    y : int64
        Current y-position index.
    z : int64
        Current z-position index.
    mesh : TYPE
        iqmc mesh.

    Returns
    -------
    dV : float64
        cell volume.

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


@njit
def iqmc_sample_position(a, b, sample):
    return a + (b - a) * sample


@njit
def iqmc_sample_isotropic_direction(sample1, sample2):
    """

    Sample the an isotropic direction using samples between [0,1].

    Parameters
    ----------
    sample1 : float64
        LDS sample 1.
    sample2 : float64
        LDS sample 2.

    Returns
    -------
    ux : float64
        x direction.
    uy : float64
        y direction.
    uz : float64
        z direction.

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


@njit
def iqmc_sample_group(sample, G):
    """
    Uniformly sample energy group using a random sample between [0,1].

    Parameters
    ----------
    sample : float64
        LDS sample.
    G : int64
        Number of energy groups.

    Returns
    -------
    int64
        Assigned energy group.

    """
    return int(np.floor(sample * G))


@njit
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
    # variables for cell finding functions
    trans = np.zeros((3,))
    # create particle to utilize cell finding functions
    P_temp = np.zeros(1, dtype=type_.particle)[0]
    # set default attributes
    P_temp["alive"] = True
    P_temp["material_ID"] = -1
    P_temp["cell_ID"] = -1

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
                    P_temp["cell_ID"] = get_particle_cell(P_temp, 0, trans, mcdc)

                    # set material_ID
                    material_ID = get_particle_material(P_temp, mcdc)

                    # assign material index
                    mcdc["technique"]["iqmc"]["material_idx"][t, i, j, k] = material_ID


@njit
def iqmc_reset_tallies(iqmc):
    score_bin = iqmc["score"]
    score_list = iqmc["score_list"]

    iqmc["source"].fill(0.0)
    for name in literal_unroll(iqmc_score_list):
        if score_list[name]:
            score_bin[name].fill(0.0)


@njit
def iqmc_distribute_tallies(iqmc):
    score_bin = iqmc["score"]
    score_list = iqmc["score_list"]

    for name in literal_unroll(iqmc_score_list):
        if score_list[name]:
            iqmc_score_reduce_bin(score_bin[name])


@njit
def iqmc_score_reduce_bin(score):
    # MPI Reduce
    buff = np.zeros_like(score)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array(score), buff, op=MPI.SUM)
    score[:] = buff


@njit
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


@njit
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
    # bilinear xy
    if score_list["tilt-xy"]:
        Q += score_bin["tilt-xy"][:, t, x, y, z] * (P["x"] - x_mid) * (P["y"] - y_mid)
    # bilinear xz
    if score_list["tilt-xz"]:
        Q += score_bin["tilt-xz"][:, t, x, y, z] * (P["x"] - x_mid) * (P["z"] - z_mid)
    # bilinear yz
    if score_list["tilt-yz"]:
        Q += score_bin["tilt-yz"][:, t, x, y, z] * (P["y"] - y_mid) * (P["z"] - z_mid)


@njit
def iqmc_distribute_sources(mcdc):
    """
    This function is meant to distribute iqmc_total_source to the relevant
    invidual source contributions, e.x. source_total -> source, source-x,
    source-y, source-z, source-xy, etc.

    Parameters
    ----------
    mcdc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    iqmc = mcdc["technique"]["iqmc"]
    total_source = iqmc["total_source"].copy()
    shape = iqmc["source"].shape
    size = iqmc["source"].size
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    Vsize = 0

    # effective sources
    # in Davidsons method we need to separate scattering and fission
    # in all other methods we can combine them into one
    if mcdc["setting"]["mode_eigenvalue"] and iqmc["eigenmode_solver"] == "davidson":
        # effective scattering
        score_bin["effective-scattering"] = np.reshape(
            total_source[Vsize : (Vsize + size)].copy(), shape
        )
        Vsize += size
        # effective fission
        score_bin["effective-fission"] = np.reshape(
            total_source[Vsize : (Vsize + size)].copy(), shape
        )
        Vsize += size
    else:
        # effective source
        iqmc["source"] = np.reshape(total_source[Vsize : (Vsize + size)].copy(), shape)
        Vsize += size

    # source tilting arrays
    tilt_list = [
        "tilt-x",
        "tilt-y",
        "tilt-z",
        "tilt-xy",
        "tilt-xz",
        "tilt-yz",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            score_bin[name] = np.reshape(total_source[Vsize : (Vsize + size)], shape)
            Vsize += size


@njit
def iqmc_consolidate_sources(mcdc):
    """
    This function is meant to collect the relevant invidual source
    contributions, e.x. source, source-x, source-y, source-z, source-xy, etc.
    and combine them into one vector (source_total)

    Parameters
    ----------
    mcdc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    iqmc = mcdc["technique"]["iqmc"]
    total_source = iqmc["total_source"]
    size = iqmc["source"].size
    score_list = iqmc["score_list"]
    score_bin = iqmc["score"]
    Vsize = 0

    # effective sources
    # in Davidsons method we need to separate scattering and fission
    # in all other methods we can combine them into one
    if mcdc["setting"]["mode_eigenvalue"] and iqmc["eigenmode_solver"] == "davidson":
        # effective scattering array
        total_source[Vsize : (Vsize + size)] = np.reshape(
            score_bin["effective-scattering"].copy(), size
        )
        Vsize += size
        # effective fission array
        total_source[Vsize : (Vsize + size)] = np.reshape(
            score_bin["effective-fission"].copy(), size
        )
        Vsize += size
    else:
        # effective source
        total_source[Vsize : (Vsize + size)] = np.reshape(iqmc["source"].copy(), size)
        Vsize += size

    # source tilting arrays
    tilt_list = [
        "tilt-x",
        "tilt-y",
        "tilt-z",
        "tilt-xy",
        "tilt-xz",
        "tilt-yz",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            total_source[Vsize : (Vsize + size)] = np.reshape(score_bin[name], size)
            Vsize += size


# =============================================================================
# iQMC Tallies
# =============================================================================
# TODO: Not all ST tallies have been built for case where SigmaT = 0.0


@njit
def iqmc_flux(SigmaT, w, distance, dV):
    # Score Flux
    if SigmaT.all() > 0.0:
        return w * (1 - np.exp(-(distance * SigmaT))) / (SigmaT * dV)
    else:
        return distance * w / dV


@njit
def iqmc_fission_source(phi, material):
    SigmaF = material["fission"]
    nu_f = material["nu_f"]
    return np.sum(nu_f * SigmaF * phi)


@njit
def iqmc_fission_power(phi, material):
    SigmaF = material["fission"]
    return SigmaF * phi


@njit
def iqmc_effective_fission(phi, mat_id, mcdc):
    """
    Calculate the fission source for use with iQMC.

    Parameters
    ----------
    phi : float64
        scalar flux in the spatial cell
    mat_idx :
        material index
    mcdc : TYPE
        DESCRIPTION.

    Returns
    -------
    float64
        fission source

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


@njit
def iqmc_effective_scattering(phi, mat_id, mcdc):
    """
    Calculate the scattering source for use with iQMC.

    Parameters
    ----------
    phi : float64
        scalar flux in the spatial cell
    mat_idx :
        material index
    mcdc : TYPE
        DESCRIPTION.

    Returns
    -------
    float64
        scattering source

    """
    material = mcdc["materials"][mat_id]
    chi_s = material["chi_s"]
    SigmaS = material["scatter"]
    return np.dot(chi_s.T, SigmaS * phi)


@njit
def iqmc_effective_source(phi, mat_id, mcdc):
    S = iqmc_effective_scattering(phi, mat_id, mcdc)
    F = iqmc_effective_fission(phi, mat_id, mcdc)
    return S + F


@njit
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


@njit
def iqmc_bilinear_tilt(ux, x, dx, x_mid, uy, y, dy, y_mid, dt, dz, w, S, SigmaT):
    # TODO: integral incase of SigmaT = 0
    Q = (
        (1 / SigmaT**3)
        * w
        * (
            (x - x_mid) * SigmaT * (uy + (y - y_mid) * SigmaT)
            + ux * (2 * uy + (y - y_mid) * SigmaT)
            + np.exp(-S * SigmaT)
            * (
                -2 * ux * uy
                + ((-x + x_mid) * uy + ux * (-y + y_mid - 2 * S * uy)) * SigmaT
                - (x - x_mid + S * ux) * (y - y_mid + S * uy) * SigmaT**2
            )
        )
    )

    Q *= 144 / (dt * dx**3 * dy**3 * dz)
    return Q


# =============================================================================
# iQMC Iterative Method Mapping Functions
# =============================================================================


@njit
def AxV(V, b, mcdc):
    """
    Linear operator to be used with GMRES.
    Calculate action of A on input vector V, where A is a transport sweep
    and V is the total source (constant and tilted).
    """
    iqmc = mcdc["technique"]["iqmc"]
    iqmc["total_source"] = V.copy()
    # distribute segments of V to appropriate sources
    iqmc_distribute_sources(mcdc)
    # reset bank size
    mcdc["bank_source"]["size"] = 0
    # QMC Sweep
    iqmc_prepare_particles(mcdc)
    iqmc_reset_tallies(iqmc)
    iqmc["sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_tallies(iqmc)
    # update source adds effective scattering + fission + fixed-source
    iqmc_update_source(mcdc)
    # combine all sources (constant and tilted) into one vector
    iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()
    axv = V - (v_out - b)

    return axv


@njit
def HxV(V, mcdc):
    """
    Linear operator for Davidson method,
    scattering + streaming terms -> (I-L^(-1)S)*phi
    """
    iqmc = mcdc["technique"]["iqmc"]
    # flux input is most recent iteration of eigenvector
    v = V[:, -1]
    iqmc["total_source"] = v.copy()
    iqmc_distribute_sources(mcdc)
    # reset bank size
    mcdc["bank_source"]["size"] = 0

    # QMC Sweep
    # prepare_qmc_scattering_source(mcdc)
    iqmc["source"] = iqmc["fixed_source"] + iqmc["score"]["effective-scattering"]
    iqmc_prepare_particles(mcdc)
    iqmc_reset_tallies(iqmc)
    iqmc["sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_tallies(iqmc)
    iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()
    axv = v - v_out

    return axv


@njit
def FxV(V, mcdc):
    """
    Linear operator for Davidson method,
    fission term -> (L^(-1)F*phi)
    """
    iqmc = mcdc["technique"]["iqmc"]
    # flux input is most recent iteration of eigenvector
    v = V[:, -1]
    # reshape v and assign to iqmc_flux
    iqmc["total_source"] = v.copy()
    iqmc_distribute_sources(mcdc)
    # reset bank size
    mcdc["bank_source"]["size"] = 0

    # QMC Sweep
    iqmc["source"] = iqmc["fixed_source"] + iqmc["score"]["effective-fission"]
    iqmc_prepare_particles(mcdc)
    iqmc_reset_tallies(iqmc)
    iqmc["sweep_counter"] += 1
    loop_source(0, mcdc)

    # sum resultant flux on all processors
    iqmc_distribute_tallies(iqmc)
    iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()

    return v_out


@njit
def preconditioner(V, mcdc, num_sweeps=3):
    """
    Linear operator approximation of (I-L^(-1)S)*phi

    In this case the preconditioner is a specified number of purely scattering
    transport sweeps.
    """
    iqmc = mcdc["technique"]["iqmc"]
    # flux input is most recent iteration of eigenvector
    iqmc["total_source"] = V.copy()
    iqmc_distribute_sources(mcdc)

    for i in range(num_sweeps):
        # reset bank size
        mcdc["bank_source"]["size"] = 0
        # QMC Sweep
        iqmc["source"] = iqmc["fixed_source"] + iqmc["score"]["effective-scattering"]
        iqmc_prepare_particles(mcdc)
        iqmc_reset_tallies(iqmc)
        iqmc["sweep_counter"] += 1
        loop_source(0, mcdc)
        # sum resultant flux on all processors
        iqmc_distribute_tallies(iqmc)

    iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()
    v_out = V - v_out

    return v_out


# =============================================================================
# Weight Roulette
# =============================================================================


@njit
def weight_roulette(P, mcdc):
    """
    If neutron weight below wr_threshold, then enter weight rouelette
    technique. Neutron has 'chance' probability of having its weight increased
    by factor of 1/CHANCE, and 1-CHANCE probability of terminating.

    Parameters
    ----------
    P :
    mcdc :

    Returns
    -------
    None.

    """
    chance = mcdc["technique"]["wr_chance"]
    x = rng(P)
    if x <= chance:
        P["iqmc"]["w"] /= chance
        P["w"] /= chance
    else:
        P["alive"] = False


# ==============================================================================
# Sensitivity quantification (Derivative Source Method)
# =============================================================================


@njit
def sensitivity_surface(P, surface, material_ID_old, material_ID_new, mcdc):
    # Sample number of derivative sources
    xi = surface["dsm_Np"]
    if xi != 1.0:
        Np = int(math.floor(xi + rng(P)))
    else:
        Np = 1

    # Terminate and put the current particle into the secondary bank
    P["alive"] = False
    add_particle(copy_particle(P), mcdc["bank_active"])

    # Get sensitivity ID
    ID = surface["sensitivity_ID"]
    if mcdc["technique"]["dsm_order"] == 2:
        ID1 = min(P["sensitivity_ID"], ID)
        ID2 = max(P["sensitivity_ID"], ID)
        ID = get_DSM_ID(ID1, ID2, mcdc["setting"]["N_sensitivity"])

    # Get materials
    material_old = mcdc["materials"][material_ID_old]
    material_new = mcdc["materials"][material_ID_new]

    # Determine the plus and minus components and then their weight signs
    trans = P["translation"]
    sign_origin = surface_normal_component(P, surface, trans)
    if sign_origin > 0.0:
        # New is +, old is -
        sign_new = -1.0
        sign_old = 1.0
    else:
        sign_new = 1.0
        sign_old = -1.0

    # Get XS
    g = P["g"]
    SigmaT_old = material_old["total"][g]
    SigmaT_new = material_new["total"][g]
    SigmaS_old = material_old["scatter"][g]
    SigmaS_new = material_new["scatter"][g]
    SigmaF_old = material_old["fission"][g]
    SigmaF_new = material_new["fission"][g]
    nu_s_old = material_old["nu_s"][g]
    nu_s_new = material_new["nu_s"][g]
    nu_old = material_old["nu_f"][g]
    nu_new = material_new["nu_f"][g]
    nuSigmaS_old = nu_s_old * SigmaS_old
    nuSigmaS_new = nu_s_new * SigmaS_new
    nuSigmaF_old = nu_old * SigmaF_old
    nuSigmaF_new = nu_new * SigmaF_new

    # Get source type probabilities
    delta = -(SigmaT_old * sign_old + SigmaT_new * sign_new)
    scatter = nuSigmaS_old * sign_old + nuSigmaS_new * sign_new
    fission = nuSigmaF_old * sign_old + nuSigmaF_new * sign_new
    p_delta = abs(delta)
    p_scatter = abs(scatter)
    p_fission = abs(fission)
    p_total = p_delta + p_scatter + p_fission

    # Get inducing flux
    #   Apply constant flux approximation for tangent direction
    #   [Dupree 2002, Eq. (7.39)]
    mu = abs(sign_origin)
    epsilon = 0.01
    if mu < epsilon:
        mu = epsilon / 2
    flux = P["w"] / mu

    # Base weight
    w_hat = p_total * flux / xi

    # Sample the derivative sources
    for n in range(Np):
        # Create new particle
        P_new = split_particle(P)

        # Sample source type
        xi = rng(P) * p_total
        tot = p_delta
        if tot > xi:
            # Delta source
            sign_delta = delta / p_delta
            P_new["w"] = w_hat * sign_delta
        else:
            tot += p_scatter
            if tot > xi:
                # Scattering source
                total_scatter = nuSigmaS_old + nuSigmaS_new
                w_s = w_hat * total_scatter / p_scatter

                # Sample if it is from + or - component
                if nuSigmaS_old > rng(P) * total_scatter:
                    sample_phasespace_scattering(P, material_old, P_new, mcdc)
                    P_new["w"] = w_s * sign_old
                else:
                    sample_phasespace_scattering(P, material_new, P_new, mcdc)
                    P_new["w"] = w_s * sign_new
            else:
                # Fission source
                total_fission = nuSigmaF_old + nuSigmaF_new
                w_f = w_hat * total_fission / p_fission

                # Sample if it is from + or - component
                if nuSigmaF_old > rng(P) * total_fission:
                    sample_phasespace_fission(P, material_old, P_new, mcdc)
                    P_new["w"] = w_f * sign_old
                else:
                    sample_phasespace_fission(P, material_new, P_new, mcdc)
                    P_new["w"] = w_f * sign_new

        # Assign sensitivity_ID
        P_new["sensitivity_ID"] = ID

        # Shift back if needed to ensure crossing
        sign = surface_normal_component(P_new, surface, trans)
        if sign_origin * sign > 0.0:
            # Get surface normal
            nx, ny, nz = surface_normal(P_new, surface, trans)

            # The shift
            if sign > 0.0:
                P_new["x"] -= nx * 2 * SHIFT
                P_new["y"] -= ny * 2 * SHIFT
                P_new["z"] -= nz * 2 * SHIFT
            else:
                P_new["x"] += nx * 2 * SHIFT
                P_new["y"] += ny * 2 * SHIFT
                P_new["z"] += nz * 2 * SHIFT

        # Put the current particle into the secondary bank
        add_particle(P_new, mcdc["bank_active"])

    # Sample potential second-order sensitivity particles?
    if mcdc["technique"]["dsm_order"] < 2 or P["sensitivity_ID"] > 0:
        return

    # Get total probability
    p_total = 0.0
    for material in [material_new, material_old]:
        if material["sensitivity"]:
            N_nuclide = material["N_nuclide"]
            for i in range(N_nuclide):
                nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
                if nuclide["sensitivity"]:
                    sigmaT = nuclide["total"][g]
                    sigmaS = nuclide["scatter"][g]
                    sigmaF = nuclide["fission"][g]
                    nu_s = nuclide["nu_s"][g]
                    nu = nuclide["nu_f"][g]
                    nusigmaS = nu_s * sigmaS
                    nusigmaF = nu * sigmaF
                    total = sigmaT + nusigmaS + nusigmaF
                    p_total += total

    # Base weight
    w = p_total * flux / surface["dsm_Np"]

    # Sample source
    for n in range(Np):
        source_obtained = False

        # Create new particle
        P_new = split_particle(P)

        # Sample term
        xi = rng(P_new) * p_total
        tot = 0.0
        for material_ID, sign in zip(
            [material_ID_new, material_ID_old], [sign_new, sign_old]
        ):
            material = mcdc["materials"][material_ID]
            if material["sensitivity"]:
                N_nuclide = material["N_nuclide"]
                for i in range(N_nuclide):
                    nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
                    if nuclide["sensitivity"]:
                        # Source ID
                        ID1 = min(nuclide["sensitivity_ID"], surface["sensitivity_ID"])
                        ID2 = max(nuclide["sensitivity_ID"], surface["sensitivity_ID"])
                        ID_source = get_DSM_ID(
                            ID1, ID2, mcdc["setting"]["N_sensitivity"]
                        )

                        sigmaT = nuclide["total"][g]
                        sigmaS = nuclide["scatter"][g]
                        sigmaF = nuclide["fission"][g]
                        nu_s = nuclide["nu_s"][g]
                        nu = nuclide["nu_f"][g]
                        nusigmaS = nu_s * sigmaS
                        nusigmaF = nu * sigmaF

                        tot += sigmaT
                        if tot > xi:
                            # Delta source
                            P_new["w"] = -w * sign
                            P_new["sensitivity_ID"] = ID_source
                            add_particle(P_new, mcdc["bank_active"])
                            source_obtained = True
                        else:
                            P_new["w"] = w * sign

                            tot += nusigmaS
                            if tot > xi:
                                # Scattering source
                                sample_phasespace_scattering_nuclide(
                                    P, nuclide, P_new, mcdc
                                )
                                P_new["sensitivity_ID"] = ID_source
                                add_particle(P_new, mcdc["bank_active"])
                                source_obtained = True
                            else:
                                tot += nusigmaF
                                if tot > xi:
                                    # Fission source
                                    sample_phasespace_fission_nuclide(
                                        P, nuclide, P_new, mcdc
                                    )
                                    P_new["sensitivity_ID"] = ID_source
                                    add_particle(P_new, mcdc["bank_active"])
                                    source_obtained = True
                    if source_obtained:
                        break
                if source_obtained:
                    break


@njit
def sensitivity_material(P, mcdc):
    # The incident particle is already terminated

    # Get material
    material = mcdc["materials"][P["material_ID"]]

    # Check if sensitivity nuclide is sampled
    g = P["g"]
    SigmaT = material["total"][g]
    N_nuclide = material["N_nuclide"]
    if N_nuclide == 1:
        nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
    else:
        xi = rng(P) * SigmaT
        tot = 0.0
        for i in range(N_nuclide):
            nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
            density = material["nuclide_densities"][i]
            tot += density * nuclide["total"][g]
            if xi < tot:
                break
    if not nuclide["sensitivity"]:
        return

    # Sample number of derivative sources
    xi = nuclide["dsm_Np"]
    if xi != 1.0:
        Np = int(math.floor(xi + rng(P)))
    else:
        Np = 1

    # Get sensitivity ID
    ID = nuclide["sensitivity_ID"]
    double = False
    if mcdc["technique"]["dsm_order"] == 2:
        ID1 = min(P["sensitivity_ID"], ID)
        ID2 = max(P["sensitivity_ID"], ID)
        ID = get_DSM_ID(ID1, ID2, mcdc["setting"]["N_sensitivity"])
        if ID1 == ID2:
            double = True

    # Undo implicit capture
    if mcdc["technique"]["implicit_capture"]:
        SigmaC = material["capture"][g]
        P["w"] *= SigmaT / (SigmaT - SigmaC)

    # Get XS
    g = P["g"]
    sigmaT = nuclide["total"][g]
    sigmaS = nuclide["scatter"][g]
    sigmaF = nuclide["fission"][g]
    nu_s = nuclide["nu_s"][g]
    nu = nuclide["nu_f"][g]
    nusigmaS = nu_s * sigmaS
    nusigmaF = nu * sigmaF

    # Base weight
    total = sigmaT + nusigmaS + nusigmaF
    w = total * P["w"] / sigmaT / xi

    # Double if it's self-second-order
    if double:
        w *= 2

    # Sample the derivative sources
    for n in range(Np):
        # Create new particle
        P_new = split_particle(P)

        # Sample source type
        xi = rng(P_new) * total
        tot = sigmaT
        if tot > xi:
            # Delta source
            P_new["w"] = -w
        else:
            P_new["w"] = w

            tot += nusigmaS
            if tot > xi:
                # Scattering source
                sample_phasespace_scattering_nuclide(P, nuclide, P_new, mcdc)
            else:
                # Fission source
                sample_phasespace_fission_nuclide(P, nuclide, P_new, mcdc)

        # Assign sensitivity_ID
        P_new["sensitivity_ID"] = ID

        # Put the current particle into the secondary bank
        add_particle(P_new, mcdc["bank_active"])


# ==============================================================================
# Particle tracker
# ==============================================================================


@njit
def track_particle(P, mcdc):
    idx = mcdc["particle_track_N"]
    mcdc["particle_track"][idx, 0] = mcdc["particle_track_history_ID"]
    mcdc["particle_track"][idx, 1] = mcdc["particle_track_particle_ID"]
    mcdc["particle_track"][idx, 2] = P["g"] + 1
    mcdc["particle_track"][idx, 3] = P["t"]
    mcdc["particle_track"][idx, 4] = P["x"]
    mcdc["particle_track"][idx, 5] = P["y"]
    mcdc["particle_track"][idx, 6] = P["z"]
    mcdc["particle_track"][idx, 7] = P["w"]
    mcdc["particle_track_N"] += 1


# ==============================================================================
# Derivative Source Method (DSM)
# ==============================================================================


@njit
def get_DSM_ID(ID1, ID2, Np):
    # First-order sensitivity
    if ID1 == 0:
        return ID2

    # Self second-order
    if ID1 == ID2:
        return Np + ID1

    # Cross second-order
    ID1 -= 1
    ID2 -= 1
    return int(
        2 * Np + (Np * (Np - 1) / 2) - (Np - ID1) * ((Np - ID1) - 1) / 2 + ID2 - ID1
    )


# =============================================================================
# Continuous Energy Physics
# =============================================================================


@njit
def get_MacroXS(type_, material, P, mcdc):
    # Multigroup XS
    g = P["g"]
    if mcdc["setting"]["mode_MG"]:
        if type_ == XS_TOTAL:
            return material["total"][g]
        elif type_ == XS_SCATTER:
            return material["scatter"][g]
        elif type_ == XS_CAPTURE:
            return material["capture"][g]
        elif type_ == XS_FISSION:
            return material["fission"][g]

    # Continuous-energy XS
    MacroXS = 0.0
    E = P["E"]

    # Sum over all nuclides
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]

        # Skip if not compatible
        if type_ == XS_FISSION and not nuclide["fissionable"]:
            continue

        # Get nuclide density
        N = material["nuclide_densities"][i]

        # Get microscopic cross-section
        microXS = get_microXS(type_, nuclide, E)

        # Accumulate
        MacroXS += N * microXS

    return MacroXS


@njit
def get_microXS(type_, nuclide, E):
    # Get type_ XS vector data
    if type_ == XS_TOTAL:
        data = nuclide["ce_total"]
    elif type_ == XS_SCATTER:
        data = nuclide["ce_scatter"]
    elif type_ == XS_CAPTURE:
        data = nuclide["ce_capture"]
    elif type_ == XS_FISSION:
        data = nuclide["ce_fission"]
    return get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])


@njit
def get_XS(data, E, E_grid, NE):
    # Search XS energy bin index
    idx = binary_search(E, E_grid, NE)

    # Extrapolate if E is outside the given data
    if idx == -1:
        idx = 0
    elif idx + 1 == NE:
        idx -= 1

    # Linear interpolation
    E1 = E_grid[idx]
    E2 = E_grid[idx + 1]
    XS1 = data[idx]
    XS2 = data[idx + 1]

    return XS1 + (E - E1) * (XS2 - XS1) / (E2 - E1)


@njit
def get_nu(type_, nuclide, E, group=-1):
    if type_ == NU_TOTAL:
        nu = get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])
        for i in range(6):
            nu += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return nu

    if type_ == NU_PROMPT:
        return get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])

    if type_ == NU_DELAYED and group == -1:
        tot = 0.0
        for i in range(6):
            tot += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return tot

    if type_ == NU_DELAYED and group != -1:
        return get_XS(
            nuclide["ce_nu_d"][group], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
        )


@njit
def sample_nuclide(material, P, type_, mcdc):
    xi = rng(P) * get_MacroXS(type_, material, P, mcdc)
    tot = 0.0
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]

        # Skip if not compatible
        if type_ == XS_FISSION and not nuclide["fissionable"]:
            continue

        N = material["nuclide_densities"][i]
        tot += N * get_microXS(type_, nuclide, P["E"])
        if tot > xi:
            break

    return nuclide


@njit
def sample_Eout(P_new, E_grid, NE, chi):
    xi = rng(P_new)

    # Determine bin index
    idx = binary_search(xi, chi, NE)

    # Linear interpolation
    E1 = E_grid[idx]
    E2 = E_grid[idx + 1]
    chi1 = chi[idx]
    chi2 = chi[idx + 1]
    return E1 + (xi - chi1) * (E2 - E1) / (chi2 - chi1)


# =============================================================================
# Miscellany
# =============================================================================


@njit
def binary_search(val, grid, length=0):
    """
    Binary search that returns the bin index of the value `val` given grid `grid`.

    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                                 (-1 if val = min(grid)
    """

    left = 0
    if length == 0:
        right = len(grid) - 1
    else:
        right = length - 1
    mid = -1
    while left <= right:
        mid = int((left + right) / 2)
        if grid[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return int(right)


@njit
def lartg(f, g):
    """
    Originally a Lapack routine to generate a plane rotation with
    real cosine and real sine.

    Reference
    ----------
    https://netlib.org/lapack/explore-html/df/dd1/group___o_t_h_e_rauxiliary_ga86f8f877eaea0386cdc2c3c175d9ea88.html#:~:text=DLARTG%20generates%20a%20plane%20rotation%20with%20real%20cosine,%3D%20G%20%2F%20R%20Hence%20C%20%3E%3D%200.

    Parameters
    ----------
    f :  The first component of vector to be rotated.
    g :  The second component of vector to be rotated.

    Returns
    -------
    c : The cosine of the rotation.
    s : The sine of the rotation.
    r : The nonzero component of the rotated vector.

    """
    r = np.sign(f) * np.sqrt(f * f + g * g)
    c = f / r
    s = g / r
    return c, s, r


@njit
def modified_gram_schmidt(V, u):
    """
    Modified Gram Schmidt routine

    """
    u = np.reshape(u, (u.size, 1))
    V = np.ascontiguousarray(V)
    w1 = u - np.dot(V, np.dot(V.T, u))
    v1 = w1 / np.linalg.norm(w1)
    w2 = v1 - np.dot(V, np.dot(V.T, v1))
    v2 = w2 / np.linalg.norm(w2)
    V = np.append(V, v2, axis=1)
    return V


# =============================================================================
# Variance Deconvolution
# =============================================================================


@njit
def uq_resample(mean, delta, info):
    # Currently only uniform distribution
    shape = mean.shape
    size = mean.size
    xi = rng_array(info["rng_seed"], shape, size)

    return mean + (2 * xi - 1) * delta


@njit
def reset_material(mcdc, idm, material_uq):
    # Assumes all nuclides have already been re-sampled
    # Basic XS
    material = mcdc["materials"][idm]
    for tag in literal_unroll(("capture", "scatter", "fission", "total")):
        if material_uq["flags"][tag]:
            material[tag][:] = 0.0
            for n in range(material["N_nuclide"]):
                nuc1 = mcdc["nuclides"][material["nuclide_IDs"][n]]
                density = material["nuclide_densities"][n]
                material[tag] += nuc1[tag] * density

    # Effective speed
    if material_uq["flags"]["speed"]:
        material["speed"][:] = 0.0
        for n in range(material["N_nuclide"]):
            nuc2 = mcdc["nuclides"][material["nuclide_IDs"][n]]
            density = material["nuclide_densities"][n]
            material["speed"] += nuc2["speed"] * nuc2["total"] * density
        if max(material["total"]) == 0.0:
            material["speed"][:] = nuc2["speed"][:]
        else:
            material["speed"] /= material["total"]

    # Calculate effective spectra and multiplicities of scattering and prompt fission
    G = material["G"]
    if max(material["scatter"]) > 0.0:
        shape = material["chi_s"].shape
        nuSigmaS = np.zeros(shape)
        for i in range(material["N_nuclide"]):
            nuc3 = mcdc["nuclides"][material["nuclide_IDs"][i]]
            density = material["nuclide_densities"][i]
            SigmaS = np.diag(nuc3["scatter"]) * density
            nu_s = np.diag(nuc3["nu_s"])
            chi_s = np.ascontiguousarray(nuc3["chi_s"].transpose())
            nuSigmaS += chi_s.dot(nu_s.dot(SigmaS))
        chi_nu_s = nuSigmaS.dot(np.diag(1.0 / material["scatter"]))
        material["nu_s"] = np.sum(chi_nu_s, axis=0)
        material["chi_s"] = np.ascontiguousarray(
            chi_nu_s.dot(np.diag(1.0 / material["nu_s"])).transpose()
        )
    if max(material["fission"]) > 0.0:
        nuSigmaF = np.zeros((G, G), dtype=float)
        for n in range(material["N_nuclide"]):
            nuc4 = mcdc["nuclides"][material["nuclide_IDs"][n]]
            density = material["nuclide_densities"][n]
            SigmaF = np.diag(nuc4["fission"]) * density
            nu_p = np.diag(nuc4["nu_p"])
            chi_p = np.ascontiguousarray(np.transpose(nuc4["chi_p"]))
            nuSigmaF += chi_p.dot(nu_p.dot(SigmaF))
        chi_nu_p = nuSigmaF.dot(np.diag(1.0 / material["fission"]))
        material["nu_p"] = np.sum(chi_nu_p, axis=0)
        # Required because the below function otherwise returns an F-contiguous array
        material["chi_p"] = np.ascontiguousarray(
            np.transpose(chi_nu_p.dot(np.diag(1.0 / material["nu_p"])))
        )

    # Calculate delayed and total fission multiplicities
    if max(material["fission"]) > 0.0:
        material["nu_f"][:] = material["nu_p"][:]
        for j in range(material["J"]):
            total = np.zeros(material["G"])
            for n in range(material["N_nuclide"]):
                nuc5 = mcdc["nuclides"][material["nuclide_IDs"][n]]
                density = material["nuclide_densities"][n]
                total += nuc5["nu_d"][:, j] * nuc5["fission"] * density
            material["nu_d"][:, j] = total / material["fission"]
            material["nu_f"] += material["nu_d"][:, j]


@njit
def reset_nuclide(nuclide, nuclide_uq):
    for name in literal_unroll(
        ("speed", "decay", "capture", "fission", "nu_s", "nu_p")
    ):
        if nuclide_uq["flags"][name]:
            nuclide[name] = uq_resample(
                nuclide_uq["mean"][name], nuclide_uq["delta"][name], nuclide_uq["info"]
            )

    if nuclide_uq["flags"]["scatter"]:
        scatter = uq_resample(
            nuclide_uq["mean"]["scatter"],
            nuclide_uq["delta"]["scatter"],
            nuclide_uq["info"],
        )
        nuclide["scatter"] = np.sum(scatter, 0)
        nuclide["chi_s"][:, :] = np.swapaxes(scatter, 0, 1)[:, :]
        for g in range(nuclide["G"]):
            if nuclide["scatter"][g] > 0.0:
                nuclide["chi_s"][g, :] /= nuclide["scatter"][g]

    if nuclide_uq["flags"]["total"]:
        nuclide["total"][:] = (
            nuclide["capture"] + nuclide["scatter"] + nuclide["fission"]
        )

    if nuclide_uq["flags"]["nu_d"]:
        nu_d = uq_resample(
            nuclide_uq["mean"]["nu_d"], nuclide_uq["delta"]["nu_d"], nuclide_uq["info"]
        )
        nuclide["nu_d"][:, :] = np.swapaxes(nu_d, 0, 1)[:, :]

    if nuclide_uq["flags"]["nu_f"]:  # True if either nu_p or nu_d is true
        nuclide["nu_f"] = nuclide["nu_p"]
        for j in range(nuclide["J"]):
            nuclide["nu_f"] += nuclide["nu_d"][:, j]

    # Prompt fission spectrum (If G == 1, all ones)
    if nuclide_uq["flags"]["chi_p"]:
        chi_p = uq_resample(
            nuclide_uq["mean"]["chi_p"],
            nuclide_uq["delta"]["chi_p"],
            nuclide_uq["info"],
        )
        nuclide["chi_p"][:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
        # Normalize
        for g in range(nuclide["G"]):
            if np.sum(nuclide["chi_p"][g, :]) > 0.0:
                nuclide["chi_p"][g, :] /= np.sum(nuclide["chi_p"][g, :])

    # Delayed fission spectrum (matrix of size JxG)
    if nuclide_uq["flags"]["chi_d"]:
        chi_d = uq_resample(
            nuclide_uq["mean"]["chi_d"],
            nuclide_uq["delta"]["chi_d"],
            nuclide_uq["info"],
        )
        # Transpose: [gout, dg] -> [dg, gout]
        nuclide["chi_d"][:, :] = np.swapaxes(chi_d, 0, 1)[:, :]
        # Normalize
        for dg in range(nuclide["J"]):
            if np.sum(nuclide["chi_d"][dg, :]) > 0.0:
                nuclide["chi_d"][dg, :] /= np.sum(nuclide["chi_d"][dg, :])


@njit
def uq_reset(mcdc, seed):
    # Types of uq parameters: materials, nuclides
    N = len(mcdc["technique"]["uq_"]["nuclides"])
    for i in range(N):
        mcdc["technique"]["uq_"]["nuclides"][i]["info"]["rng_seed"] = split_seed(
            i, seed
        )
        idn = mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"]
        reset_nuclide(mcdc["nuclides"][idn], mcdc["technique"]["uq_"]["nuclides"][i])

    M = len(mcdc["technique"]["uq_"]["materials"])
    for i in range(M):
        mcdc["technique"]["uq_"]["materials"][i]["info"]["rng_seed"] = split_seed(
            i, seed
        )
        idm = mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"]
        reset_material(mcdc, idm, mcdc["technique"]["uq_"]["materials"][i])


@njit
def uq_tally_closeout_history(mcdc):
    tally = mcdc["tally"]
    uq_tally = mcdc["technique"]["uq_tally"]

    for name in literal_unroll(score_list):
        if uq_tally[name]:
            uq_score_closeout_history(tally["score"][name], uq_tally["score"][name])


@njit
def uq_score_closeout_history(score, uq_score):
    # Assumes N_batch > 1
    # Accumulate square of history score, but continue to accumulate bin
    history_bin = score["bin"] - uq_score["batch_bin"]
    uq_score["batch_var"][:] += history_bin**2
    uq_score["batch_bin"] = score["bin"]


@njit
def uq_tally_closeout_batch(mcdc):
    uq_tally = mcdc["technique"]["uq_tally"]

    for name in literal_unroll(score_list):
        if uq_tally[name]:
            # Reset bin
            uq_tally["score"][name]["batch_bin"].fill(0.0)
            uq_reduce_bin(uq_tally["score"][name])


@njit
def uq_reduce_bin(score):
    # MPI Reduce
    buff = np.zeros_like(score["batch_var"])
    with objmode():
        MPI.COMM_WORLD.Reduce(np.array(score["batch_var"]), buff, MPI.SUM, 0)
    score["batch_var"][:] = buff


@njit
def uq_tally_closeout(mcdc):
    tally = mcdc["tally"]
    uq_tally = mcdc["technique"]["uq_tally"]

    for name in literal_unroll(score_list):
        # Uq_tally implies tally, but tally does not imply uq_tally
        if uq_tally[name]:
            uq_score_closeout(name, mcdc)
        elif tally[name]:
            score_closeout(tally["score"][name], mcdc)


@njit
def uq_score_closeout(name, mcdc):
    score = mcdc["tally"]["score"][name]
    uq_score = mcdc["technique"]["uq_tally"]["score"][name]

    N_history = mcdc["setting"]["N_particle"]

    # At this point, score["sdev"] is still just the sum of the squared mean from every batch
    uq_score["batch_var"] = (uq_score["batch_var"] / N_history - score["sdev"]) / (
        N_history - 1
    )

    # If we're here, N_batch > 1
    N_history = mcdc["setting"]["N_batch"]

    # Store results
    score["mean"][:] = score["mean"] / N_history
    uq_score["batch_var"] /= N_history
    uq_score["batch_bin"] = (score["sdev"] - N_history * np.square(score["mean"])) / (
        N_history - 1
    )
    score["sdev"][:] = np.sqrt(
        (score["sdev"] / N_history - np.square(score["mean"])) / (N_history - 1)
    )
