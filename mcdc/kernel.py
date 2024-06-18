import math

from mpi4py import MPI
from numba import njit, objmode, literal_unroll
import numba

import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import print_error, print_msg
from mcdc.type_ import iqmc_score_list
from mcdc.loop import loop_source
import mcdc.adapt as adapt
from mcdc.adapt import toggle, for_cpu, for_gpu

# =============================================================================
# Domain Decomposition
# =============================================================================

# =============================================================================
# Domain crossing event
# =============================================================================


@toggle("domain_decomp")
def domain_crossing(P, mcdc):
    # Domain mesh crossing
    seed = P["rng_seed"]
    max_size = mcdc["technique"]["dd_exchange_rate"]
    if mcdc["technique"]["domain_decomposition"]:
        mesh = mcdc["technique"]["dd_mesh"]
        # Determine which dimension is crossed
        x, y, z, t, directions = mesh_crossing_evaluate(P, mesh)
        if len(directions) == 0:
            return
        elif len(directions) > 1:
            for direction in directions[1:]:
                if direction == MESH_X:
                    P["x"] -= SHIFT * P["ux"] / np.abs(P["ux"])
                if direction == MESH_Y:
                    P["y"] -= SHIFT * P["uy"] / np.abs(P["uy"])
                if direction == MESH_Z:
                    P["z"] -= SHIFT * P["uz"] / np.abs(P["uz"])
        flag = directions[0]
        # Score on tally
        if flag == MESH_X and P["ux"] > 0:
            add_particle(P, mcdc["domain_decomp"]["bank_xp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xp"]) == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_X and P["ux"] < 0:
            add_particle(P, mcdc["domain_decomp"]["bank_xn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xn"]) == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Y and P["uy"] > 0:
            add_particle(P, mcdc["domain_decomp"]["bank_yp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yp"]) == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Y and P["uy"] < 0:
            add_particle(P, mcdc["domain_decomp"]["bank_yn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yn"]) == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Z and P["uz"] > 0:
            add_particle(P, mcdc["domain_decomp"]["bank_zp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zp"]) == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Z and P["uz"] < 0:
            add_particle(P, mcdc["domain_decomp"]["bank_zn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zn"]) == max_size:
                dd_particle_send(mcdc)
        P["alive"] = False


# =============================================================================
# Send full domain bank
# =============================================================================


requests = []


def save_request(req_pair):
    global requests

    updated_requests = []

    status = MPI.Status()
    for req, buf in requests:
        if not req.Test(status):
            updated_requests.append((req, buf))

    updated_requests.append(req_pair)
    requests = updated_requests


def clear_requests():
    global requests
    for req, buf in requests:
        req.Free()

    requests = []


@njit
def dd_check_halt(mcdc):
    return mcdc["domain_decomp"]["work_done"]


@njit
def dd_check_in(mcdc):
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0
    mcdc["domain_decomp"]["send_total"] = 0
    mcdc["domain_decomp"]["rank_busy"] = True

    with objmode(rank="int64", total="int64"):
        rank = MPI.COMM_WORLD.Get_rank()
        total = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        mcdc["domain_decomp"]["busy_total"] = total
    else:
        mcdc["domain_decomp"]["busy_total"] = 0


@njit
def dd_check_out(mcdc):
    with objmode():
        rank = MPI.COMM_WORLD.Get_rank()
        send_count = mcdc["domain_decomp"]["send_count"]
        recv_count = mcdc["domain_decomp"]["recv_count"]
        send_total = mcdc["domain_decomp"]["send_total"]
        busy_total = mcdc["domain_decomp"]["busy_total"]
        rank_busy = mcdc["domain_decomp"]["rank_busy"]

        if send_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send count {send_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_count"] = 0

        if recv_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero recv count {recv_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["recv_count"] = 0

        if send_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send total {send_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_total"] = 0

        if busy_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero busy total {busy_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["busy_total"] = 0

        if rank_busy:
            print(
                f"Domain decomposed loop closed out with rank {rank} still marked as busy"
            )
            mcdc["domain_decomp"]["rank_busy"] = 0

        clear_requests()


@njit
def dd_signal_halt(mcdc):

    with objmode():
        for rank in range(1, MPI.COMM_WORLD.Get_size()):
            dummy_buff = np.zeros((1,), dtype=np.int32)
            MPI.COMM_WORLD.Send(dummy_buff, dest=rank, tag=3)

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_signal_block(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )
    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] -= 1
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = -1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))

    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0

    if (
        (rank == 0)
        and (mcdc["domain_decomp"]["busy_total"] == 0)
        and (mcdc["domain_decomp"]["send_total"] == 0)
    ):
        dd_signal_halt(mcdc)


@njit
def dd_signal_unblock(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )

    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] += 1
        if (mcdc["domain_decomp"]["busy_total"] == 0) and (
            mcdc["domain_decomp"]["send_total"] == 0
        ):
            dd_signal_halt(mcdc)
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = 1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0


@njit
def dd_distribute_bank(mcdc, bank, dest_list):

    with objmode(send_delta="int64"):
        dest_count = len(dest_list)
        send_delta = 0
        for i, dest in enumerate(dest_list):
            size = get_bank_size(bank)
            ratio = int(size / dest_count)
            start = ratio * i
            end = start + ratio
            if i == dest_count - 1:
                end = size
            sub_bank = np.array(bank["particles"][start:end])
            if sub_bank.shape[0] > 0:
                req = MPI.COMM_WORLD.Isend(
                    [sub_bank, type_.particle_record_mpi], dest=dest, tag=1
                )
                save_request((req, sub_bank))
                send_delta += end - start

    mcdc["domain_decomp"]["send_count"] += send_delta
    set_bank_size(bank, 0)


@njit
def dd_particle_send(mcdc):
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xp"], mcdc["technique"]["dd_xp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xn"], mcdc["technique"]["dd_xn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yp"], mcdc["technique"]["dd_yp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yn"], mcdc["technique"]["dd_yn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zp"], mcdc["technique"]["dd_zp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zn"], mcdc["technique"]["dd_zn_neigh"]
    )


# =============================================================================
# Receive particles and clear banks
# =============================================================================


@njit
def dd_get_recv_tag():

    with objmode(tag="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Probe(status=status)
        tag = status.Get_tag()

    return tag


@njit
def dd_recv_particles(mcdc):

    buff = np.zeros(
        mcdc["domain_decomp"]["bank_zp"]["particles"].shape[0],
        dtype=type_.particle_record,
    )

    with objmode(size="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Recv([buff, type_.particle_record_mpi], status=status)
        size = status.Get_count(type_.particle_record_mpi)
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["recv_count"] += size

    # Set source bank from buffer
    for i in range(size):
        add_particle(buff[i], mcdc["bank_active"])

    if (
        mcdc["domain_decomp"]["recv_count"] > 0
        and not mcdc["domain_decomp"]["rank_busy"]
    ):
        dd_signal_unblock(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = True


@njit
def dd_recv_turnstile(mcdc):

    with objmode(busy_delta="int64", send_delta="int64"):
        event_buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
        MPI.COMM_WORLD.Recv([event_buff, type_.dd_turnstile_event_mpi])
        busy_delta = event_buff[0]["busy_delta"]
        send_delta = event_buff[0]["send_delta"]
        rank = MPI.COMM_WORLD.Get_rank()
        busy_total = mcdc["domain_decomp"]["busy_total"]
        send_total = mcdc["domain_decomp"]["send_total"]

    mcdc["domain_decomp"]["busy_total"] += busy_delta
    mcdc["domain_decomp"]["send_total"] += send_delta

    if (mcdc["domain_decomp"]["busy_total"] == 0) and (
        mcdc["domain_decomp"]["send_total"] == 0
    ):
        dd_signal_halt(mcdc)


@njit
def dd_recv_halt(mcdc):

    with objmode():
        dummy_buff = np.zeros((1,), dtype=np.int32)
        MPI.COMM_WORLD.Recv(dummy_buff)
        work_done = 1
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_recv(mcdc):

    if mcdc["domain_decomp"]["rank_busy"]:
        dd_signal_block(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = False

    if not mcdc["domain_decomp"]["work_done"]:
        tag = dd_get_recv_tag()

        if tag == 1:
            dd_recv_particles(mcdc)
        elif tag == 2:
            dd_recv_turnstile(mcdc)
        elif tag == 3:
            dd_recv_halt(mcdc)


# =============================================================================
# Particle in domain
# =============================================================================


# Check if particle is in domain
@njit
def particle_in_domain(P, mcdc):
    d_idx = mcdc["dd_idx"]
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    x_cell = binary_search(P["x"], mcdc["technique"]["dd_mesh"]["x"])
    y_cell = binary_search(P["y"], mcdc["technique"]["dd_mesh"]["y"])
    z_cell = binary_search(P["z"], mcdc["technique"]["dd_mesh"]["z"])

    if d_ix == x_cell:
        if d_iy == y_cell:
            if d_iz == z_cell:
                return True
    return False


# =============================================================================
# Source in domain
# =============================================================================


# Check for source in domain
@njit
def source_in_domain(source, domain_mesh, d_idx):
    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    if (
        d_x[0] <= source["box_x"][0] <= d_x[1]
        or d_x[0] <= source["box_x"][1] <= d_x[1]
        or (source["box_x"][0] < d_x[0] and source["box_x"][1] > d_x[1])
    ):
        if (
            d_y[0] <= source["box_y"][0] <= d_y[1]
            or d_y[0] <= source["box_y"][1] <= d_y[1]
            or (source["box_y"][0] < d_y[0] and source["box_y"][1] > d_y[1])
        ):
            if (
                d_z[0] <= source["box_z"][0] <= d_z[1]
                or d_z[0] <= source["box_z"][1] <= d_z[1]
                or (source["box_z"][0] < d_z[0] and source["box_z"][1] > d_z[1])
            ):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


# =============================================================================
# Compute domain load
# =============================================================================


@njit
def domain_work(mcdc, domain, N):
    domain_mesh = mcdc["technique"]["dd_mesh"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1
    work_start = 0
    for d_idx in range(domain):
        d_iz = int(d_idx / (d_Nx * d_Ny))
        d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
        d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

        d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
        d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
        d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
        # Compute volumes of sources and numbers of particles

        Psum = 0

        Nm = 0
        num_source = 0
        for source in mcdc["sources"]:
            Psum += source["prob"]
            num_source += 1
        Vi = np.zeros(num_source)
        Vim = np.zeros(num_source)
        Ni = np.zeros(num_source)
        i = 0
        for source in mcdc["sources"]:
            Ni[i] = N * source["prob"] / Psum
            Vi[i] = 1
            Vim[i] = 1
            if source["box"] == True:
                xV = source["box_x"][1] - source["box_x"][0]
                if xV != 0:
                    Vi[i] *= xV
                    Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                        source["box_x"][0], d_x[0]
                    )
                yV = source["box_y"][1] - source["box_y"][0]
                if yV != 0:
                    Vi[i] *= yV
                    Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                        source["box_y"][0], d_y[0]
                    )
                zV = source["box_z"][1] - source["box_z"][0]
                if zV != 0:
                    Vi[i] *= zV
                    Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                        source["box_z"][0], d_z[0]
                    )
            if not source_in_domain(source, domain_mesh, d_idx):
                Vim[i] = 0
            i += 1
        for source in range(num_source):
            Nm += Ni[source] * Vim[source] / Vi[source]
        work_start += Nm
    d_idx = domain
    d_iz = int(mcdc["dd_idx"] / (d_Nx * d_Ny))
    d_iy = int((mcdc["dd_idx"] - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(mcdc["dd_idx"] - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
    # Compute volumes of sources and numbers of particles
    num_source = len(mcdc["sources"])
    Vi = np.zeros(num_source)
    Vim = np.zeros(num_source)
    Ni = np.zeros(num_source)
    Psum = 0

    Nm = 0
    for source in mcdc["sources"]:
        Psum += source["prob"]
    i = 0
    for source in mcdc["sources"]:
        Ni[i] = N * source["prob"] / Psum
        Vi[i] = 1
        Vim[i] = 1

        if source["box"] == True:
            xV = source["box_x"][1] - source["box_x"][0]
            if xV != 0:
                Vi[i] *= xV
                Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                    source["box_x"][0], d_x[0]
                )
            yV = source["box_y"][1] - source["box_y"][0]
            if yV != 0:
                Vi[i] *= yV
                Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                    source["box_y"][0], d_y[0]
                )
            zV = source["box_z"][1] - source["box_z"][0]
            if zV != 0:
                Vi[i] *= zV
                Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                    source["box_z"][0], d_z[0]
                )
        i += 1
    for source in range(num_source):
        Nm += Ni[source] * Vim[source] / Vi[source]
    Nm /= mcdc["technique"]["dd_work_ratio"][domain]
    rank = mcdc["mpi_rank"]
    if mcdc["technique"]["dd_work_ratio"][domain] > 1:
        work_start += Nm * (rank - np.sum(mcdc["technique"]["dd_work_ratio"][0:d_idx]))
    total_v = 0
    for source in range(len(mcdc["sources"])):
        total_v += Vim[source]
    i = 0
    for source in mcdc["sources"]:
        if total_v != 0:
            source["prob"] *= 2 * Vim[i] / total_v
        i += 1
    return (int(Nm), int(work_start))


# =============================================================================
# Source particle in domain only
# =============================================================================


@njit()
def source_particle_dd(seed, mcdc):
    domain_mesh = mcdc["technique"]["dd_mesh"]
    d_idx = mcdc["dd_idx"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz = int(mcdc["dd_idx"] / (d_Nx * d_Ny))
    d_iy = int((mcdc["dd_idx"] - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(mcdc["dd_idx"] - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    P = np.zeros(1, dtype=type_.particle_record)[0]

    P["rng_seed"] = seed
    # Sample source
    xi = rng(P)
    tot = 0.0
    for source in mcdc["sources"]:
        if source_in_domain(source, domain_mesh, d_idx):
            tot += source["prob"]
            if tot >= xi:
                break

    # Position
    if source["box"]:
        x = sample_uniform(
            max(source["box_x"][0], d_x[0]), min(source["box_x"][1], d_x[1]), P
        )
        y = sample_uniform(
            max(source["box_y"][0], d_y[0]), min(source["box_y"][1], d_y[1]), P
        )
        z = sample_uniform(
            max(source["box_z"][0], d_z[0]), min(source["box_z"][1], d_z[1]), P
        )

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
    g = sample_discrete(source["group"], P)
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
    P["w"] = 1
    P["sensitivity_ID"] = 0
    return P


@njit
def distribute_work_dd(N, mcdc, precursor=False):
    # Total # of work
    work_size_total = N

    if not mcdc["technique"]["dd_repro"]:
        work_size, work_start = domain_work(mcdc, mcdc["dd_idx"], N)
    else:
        work_start = 0
        work_size = work_size_total

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total


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


@njit
def wrapping_mul(a, b):
    return a * b


@njit
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


@njit
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


@njit
def rng_(seed):
    seed = numba.uint64(seed)
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
    P: type_.particle_record = adapt.local_particle_record()
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
def get_bank_size(bank):
    return bank["size"][0]


@njit
def set_bank_size(bank, value):
    bank["size"][0] = value


@njit
def add_bank_size(bank, value):
    return adapt.global_add(bank["size"], 0, value)


@for_cpu()
def full_bank_print(bank):
    with objmode():
        print_error("Particle %s bank is full." % bank["tag"])


@for_gpu()
def full_bank_print(bank):
    pass


@njit
def add_particle(P, bank):

    idx = add_bank_size(bank, 1)

    # Check if bank is full
    if idx >= bank["particles"].shape[0]:
        full_bank_print(bank)

    # Set particle
    copy_recordlike(bank["particles"][idx], P)


@njit
def get_particle(P, bank, mcdc):

    idx = add_bank_size(bank, -1) - 1

    # Check if bank is empty
    if idx < 0:
        return False
        # with objmode():
        #    print_error("Particle %s bank is empty." % bank["tag"])

    # Set attribute
    P_rec = bank["particles"][idx]
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
    return True


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
        size = get_bank_size(mcdc["bank_census"])
        set_bank_size(mcdc["bank_source"], size)
        mcdc["bank_source"]["particles"][:size] = mcdc["bank_census"]["particles"][
            :size
        ]

    # MPI rebalance
    if not mcdc["technique"]["domain_decomposition"]:
        bank_rebalance(mcdc)

    # Zero out census bank
    set_bank_size(mcdc["bank_census"], 0)

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
        Nn = get_bank_size(mcdc["technique"]["IC_bank_neutron_local"])
        Np = get_bank_size(mcdc["technique"]["IC_bank_precursor_local"])

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
        start_n = add_bank_size(mcdc["technique"]["IC_bank_neutron"], Nn)
        start_p = add_bank_size(mcdc["technique"]["IC_bank_precursor"], Np)
        for i in range(Nn):
            mcdc["technique"]["IC_bank_neutron"]["particles"][start_n + i] = buff_n[i]
        for i in range(Np):
            mcdc["technique"]["IC_bank_precursor"]["precursors"][start_p + i] = buff_p[
                i
            ]

    # Reset local banks
    set_bank_size(mcdc["technique"]["IC_bank_neutron_local"], 0)
    set_bank_size(mcdc["technique"]["IC_bank_precursor_local"], 0)


@njit
def bank_scanning(bank, mcdc):
    N_local = get_bank_size(bank)

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
    N_local = get_bank_size(bank)
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
    N_DNP_local = get_bank_size(bank)

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
    for i in range(get_bank_size(bank)):
        W_local[0] += bank["particles"][i]["w"]

    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]


@njit
def allreduce(value):
    total = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array([value], np.float64), total, MPI.SUM)
    return total[0]


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
        size = get_bank_size(mcdc["bank_source"])
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
    set_bank_size(mcdc["bank_source"], size)
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


@for_cpu()
def pn_over_one():
    with objmode():
        print_error("Pn > 1.0.")


@for_gpu()
def pn_over_one():
    pass


@for_cpu()
def pp_over_one():
    with objmode():
        print_error("Pp > 1.0.")


@for_gpu()
def pp_over_one():
    pass


@njit
def bank_IC(P, prog):

    mcdc = adapt.device(prog)

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
        pn_over_one()

    # Sample particle
    if rng(P) < Pn:
        P_new = split_particle(P)
        P_new["w"] = 1.0
        P_new["t"] = 0.0
        adapt.add_IC(P_new, prog)

        # Accumulate fission
        SigmaF = material["fission"][g]
        # mcdc["technique"]["IC_fission_score"][0] += v * SigmaF
        adapt.global_add(mcdc["technique"]["IC_fission_score"], 0, v * SigmaF)

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
        pp_over_one()

    # Sample precursor
    if rng(P) < Pp:
        idx = add_bank_size(mcdc["technique"]["IC_bank_precursor_local"], 1)
        precursor = mcdc["technique"]["IC_bank_precursor_local"]["precursors"][idx]
        precursor["x"] = P["x"]
        precursor["y"] = P["y"]
        precursor["z"] = P["z"]
        precursor["w"] = wp_prime / wn_prime

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
#       The challenge is in the use of type-dependent copy_record which is
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
    set_bank_size(bank_source, 0)
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx = math.floor(tooth) - idx_start
        P = copy_record(bank_census["particles"][idx])
        # Set weight
        P["w"] *= td
        adapt.add_source(P, mcdc)


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
    set_bank_size(bank_source, 0)
    idx = 0
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx += binary_search(tooth, w_cdf[idx:])
        P = copy_record(bank_census["particles"][idx])
        # Set weight
        P["w"] = td
        adapt.add_source(P, mcdc)


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


@for_cpu()
def lost_particle(P):
    with objmode():
        print("A particle is lost at (", P["x"], P["y"], P["z"], ")")


@for_gpu()
def lost_particle(P):
    pass


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
    lost_particle(P)
    P["alive"] = False
    return -1


@njit
def get_particle_material(P, mcdc):
    # Translation accumulator
    trans_struct = adapt.local_translate()
    trans = trans_struct["values"]

    # Top level cell
    cell = mcdc["cells"][P["cell_ID"]]

    # Recursively check if cell is a lattice cell, until material cell is found
    while True:
        # Lattice cell?
        if cell["fill_type"] == FILL_LATTICE:
            # Get lattice
            lattice = mcdc["lattices"][cell["fill_ID"]]

            # Get lattice center for translation)
            for i in range(3):
                trans[i] -= cell["translation"][i]

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

    return cell["fill_ID"]


@njit
def get_particle_speed(P, mcdc):
    if mcdc["setting"]["mode_MG"]:
        return mcdc["materials"][P["material_ID"]]["speed"][P["g"]]
    else:
        return math.sqrt(P["E"]) * SQRT_E_TO_SPEED


@njit
def copy_recordlike(P_new, P):
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
    P_new["iqmc"]["w"] = P["iqmc"]["w"]
    copy_track_data(P_new, P)


@njit
def copy_record(P):
    P_new = adapt.local_particle_record()
    copy_recordlike(P_new, P)
    return P_new


@njit
def recordlike_to_particle(P_rec):
    P_new = adapt.local_particle()
    copy_recordlike(P_new, P_rec)
    P_new["fresh"] = True
    P_new["alive"] = True
    P_new["material_ID"] = -1
    P_new["cell_ID"] = -1
    P_new["surface_ID"] = -1
    P_new["event"] = -1
    return P_new


@njit
def copy_particle(P_new, P):
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["ux"] = P["ux"]
    P_new["uy"] = P["uy"]
    P_new["uz"] = P["uz"]
    P_new["g"] = P["g"]
    P_new["w"] = P["w"]
    P_new["alive"] = P["alive"]
    P_new["fresh"] = P["fresh"]
    P_new["material_ID"] = P["material_ID"]
    P_new["cell_ID"] = P["cell_ID"]
    P_new["surface_ID"] = P["surface_ID"]
    P_new["translation"] = P["translation"]
    P_new["event"] = P["event"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]
    P_new["rng_seed"] = P["rng_seed"]
    P_new["iqmc"]["w"] = P["iqmc"]["w"]
    copy_track_data(P_new, P)


@njit
def split_particle(P):
    P_new = copy_record(P)
    P_new["rng_seed"] = split_seed(P["rng_seed"], SEED_SPLIT_PARTICLE)
    rng(P)
    return P_new


# =============================================================================
# Cell operations
# =============================================================================


@njit
def cell_check(P, cell, trans, mcdc):
    region = mcdc["regions"][cell["region_ID"]]
    return region_check(P, region, trans, mcdc)


@njit
def region_check(P, region, trans, mcdc):
    if region["type"] == REGION_HALFSPACE:
        surface_ID = region["A"]
        positive_side = region["B"]

        surface = mcdc["surfaces"][surface_ID]
        side = surface_evaluate(P, surface, trans)

        if positive_side:
            if side > 0.0:
                return True
        elif side < 0.0:
            return True

        return False

    elif region["type"] == REGION_INTERSECTION:
        region_A = mcdc["regions"][region["A"]]
        region_B = mcdc["regions"][region["B"]]

        check_A = region_check(P, region_A, trans, mcdc)
        check_B = region_check(P, region_B, trans, mcdc)

        if check_A and check_B:
            return True
        else:
            return False

    elif region["type"] == REGION_COMPLEMENT:
        region_A = mcdc["regions"][region["A"]]
        if not region_check(P, region_A, trans, mcdc):
            return True
        else:
            return False

    elif region["type"] == REGION_UNION:
        region_A = mcdc["regions"][region["A"]]
        region_B = mcdc["regions"][region["B"]]

        if region_check(P, region_A, trans, mcdc):
            return True

        if region_check(P, region_B, trans, mcdc):
            return True

        return False

    elif region["type"] == REGION_ALL:
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
    if surface["BC"] == BC_VACUUM:
        P["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(P, surface, trans)


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

        div = G * ux + H * uy + I_ * uz + J1 / v
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
def mesh_get_energy_index(P, mesh, mode_MG):
    # Check if outside grid
    outside = False

    if mode_MG:
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
    x = numba.int64(math.floor((Px - mesh["x0"]) / mesh["dx"]))
    y = numba.int64(math.floor((Py - mesh["y0"]) / mesh["dy"]))
    z = numba.int64(math.floor((Pz - mesh["z0"]) / mesh["dz"]))
    return x, y, z


@njit
def mesh_crossing_evaluate(P, mesh):
    # Shift backward
    shift_particle(P, -2 * SHIFT)
    t1, x1, y1, z1, outside1 = mesh_get_index(P, mesh)

    # Double shift forward
    shift_particle(P, 4 * SHIFT)
    t2, x2, y2, z2, outside2 = mesh_get_index(P, mesh)

    # Return particle to initial position
    shift_particle(P, -2 * SHIFT)

    # Determine dimension crossed
    directions = []

    if x1 != x2:
        directions.append(MESH_X)
    if y1 != y2:
        directions.append(MESH_Y)
    if z1 != z2:
        directions.append(MESH_Z)

    return x1, y1, z1, t1, directions


# =============================================================================
# Tally operations
# =============================================================================


@njit
def score_mesh_tally(P, distance, tally, data, mcdc):
    tally_bin = data[TALLY]
    material = mcdc["materials"][P["material_ID"]]
    mesh = tally["filter"]
    stride = tally["stride"]

    # Get indices
    s = P["sensitivity_ID"]
    it, ix, iy, iz, outside = mesh_get_index(P, mesh)
    mu, azi = mesh_get_angular_index(P, mesh)
    g, outside_energy = mesh_get_energy_index(P, mesh, mcdc["setting"]["mode_MG"])

    # Outside grid?
    if outside or outside_energy:
        return

    # The tally index
    idx = (
        stride["tally"]
        + s * stride["sensitivity"]
        + mu * stride["mu"]
        + azi * stride["azi"]
        + g * stride["g"]
        + it * stride["t"]
        + ix * stride["x"]
        + iy * stride["y"]
        + iz * stride["z"]
    )

    # Score
    flux = distance * P["w"]
    for i in range(tally["N_score"]):
        score_type = tally["scores"][i]
        if score_type == SCORE_FLUX:
            score = flux
        elif score_type == SCORE_TOTAL:
            SigmaT = get_MacroXS(XS_TOTAL, material, P, mcdc)
            score = flux * SigmaT
        elif score_type == SCORE_FISSION:
            SigmaF = get_MacroXS(XS_FISSION, material, P, mcdc)
            score = flux * SigmaF
        tally_bin[TALLY_SCORE, idx + i] += score


@njit
def score_surface_tally(P, surface, tally, data, mcdc):
    # TODO: currently not supporting filters

    tally_bin = data[TALLY]
    stride = tally["stride"]

    # The tally index
    idx = stride["tally"]

    # Flux
    trans = P["translation"]
    mu = surface_normal_component(P, surface, trans)
    flux = P["w"] / abs(mu)

    # Score
    for i in range(tally["N_score"]):
        score_type = tally["scores"][i]
        if score_type == SCORE_FLUX:
            score = flux
        elif score_type == SCORE_NET_CURRENT:
            score = flux * mu
        tally_bin[TALLY_SCORE, idx + i] += score


@njit
def tally_reduce(data, mcdc):
    tally_bin = data[TALLY]
    N_bin = tally_bin.shape[1]

    # Normalize
    N_particle = mcdc["setting"]["N_particle"]
    for i in range(N_bin):
        tally_bin[TALLY_SCORE][i] /= N_particle

    # MPI Reduce
    buff = np.zeros_like(tally_bin[TALLY_SCORE])
    with objmode():
        MPI.COMM_WORLD.Reduce(tally_bin[TALLY_SCORE], buff, MPI.SUM, 0)
    tally_bin[TALLY_SCORE][:] = buff


@njit
def tally_accumulate(data, mcdc):
    tally_bin = data[TALLY]
    N_bin = tally_bin.shape[1]

    for i in range(N_bin):
        # Accumulate score and square of score into sum and sum_sq
        score = tally_bin[TALLY_SCORE, i]
        tally_bin[TALLY_SUM, i] += score
        tally_bin[TALLY_SUM_SQ, i] += score * score

        # Reset score bin
        tally_bin[TALLY_SCORE, i] = 0.0


@njit
def tally_closeout(data, mcdc):
    tally = data[TALLY]
    N_history = mcdc["setting"]["N_particle"]

    if mcdc["setting"]["N_batch"] > 1:
        N_history = mcdc["setting"]["N_batch"]

    elif mcdc["setting"]["mode_eigenvalue"]:
        N_history = mcdc["setting"]["N_active"]

    else:
        # MPI Reduce
        buff = np.zeros_like(tally[TALLY_SUM])
        buff_sq = np.zeros_like(tally[TALLY_SUM_SQ])
        with objmode():
            MPI.COMM_WORLD.Reduce(tally[TALLY_SUM], buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(tally[TALLY_SUM_SQ], buff_sq, MPI.SUM, 0)
        tally[TALLY_SUM] = buff
        tally[TALLY_SUM_SQ] = buff_sq

    # Calculate and store statistics
    #   sum --> mean
    #   sum_sq --> standard deviation
    tally[TALLY_SUM] = tally[TALLY_SUM] / N_history
    tally[TALLY_SUM_SQ] = np.sqrt(
        (tally[TALLY_SUM_SQ] / N_history - np.square(tally[TALLY_SUM]))
        / (N_history - 1)
    )


# =============================================================================
# Eigenvalue tally operations
# =============================================================================


@njit
def eigenvalue_tally(P, distance, mcdc):
    material = mcdc["materials"][P["material_ID"]]
    flux = distance * P["w"]

    # Get nu-fission
    nuSigmaF = get_MacroXS(XS_NU_FISSION, material, P, mcdc)

    # Fission production (needed even during inactive cycle)
    # mcdc["eigenvalue_tally_nuSigmaF"][0] += flux * nuSigmaF
    adapt.global_add(mcdc["eigenvalue_tally_nuSigmaF"], 0, flux * nuSigmaF)

    if mcdc["cycle_active"]:
        # Neutron density
        v = get_particle_speed(P, mcdc)
        n_density = flux / v
        # mcdc["eigenvalue_tally_n"][0] += n_density
        adapt.global_add(mcdc["eigenvalue_tally_n"], 0, n_density)
        # Maximum neutron density
        if mcdc["n_max"] < n_density:
            mcdc["n_max"] = n_density

        # Precursor density
        J = material["J"]
        SigmaF = get_MacroXS(XS_FISSION, material, P, mcdc)
        # Get the decay-wighted multiplicity
        total = 0.0
        if mcdc["setting"]["mode_MG"]:
            g = P["g"]
            for i in range(material["N_nuclide"]):
                ID_nuclide = material["nuclide_IDs"][i]
                nuclide = mcdc["nuclides"][ID_nuclide]
                for j in range(J):
                    nu_d = nuclide["nu_d"][g, j]
                    decay = nuclide["decay"][j]
                    total += nu_d / decay
        else:
            E = P["E"]
            for i in range(material["N_nuclide"]):
                ID_nuclide = material["nuclide_IDs"][i]
                nuclide = mcdc["nuclides"][ID_nuclide]
                for j in range(J):
                    nu_d = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)
                    decay = nuclide["ce_decay"][j]
                    total += nu_d / decay
        C_density = flux * total * SigmaF / mcdc["k_eff"]
        # mcdc["eigenvalue_tally_C"][0] += C_density
        adapt.global_add(mcdc["eigenvalue_tally_C"], 0, C_density)
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
            np.array(mcdc["eigenvalue_tally_nuSigmaF"]), buff_nuSigmaF, MPI.SUM
        )
        if mcdc["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_n"]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["n_max"]]), buff_nmax, MPI.MAX)
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_C"]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["C_max"]]), buff_Cmax, MPI.MAX)
            if mcdc["technique"]["IC_generator"]:
                MPI.COMM_WORLD.Allreduce(
                    np.array(mcdc["technique"]["IC_fission_score"]),
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
    mcdc["eigenvalue_tally_nuSigmaF"][0] = 0.0
    mcdc["eigenvalue_tally_n"][0] = 0.0
    mcdc["eigenvalue_tally_C"][0] = 0.0
    mcdc["technique"]["IC_fission_score"][0] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if mcdc["setting"]["gyration_radius"]:
        # Center of mass
        N_local = get_bank_size(mcdc["bank_census"])
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
def move_to_event(P, data, mcdc):
    # =========================================================================
    # Get distances to events
    # =========================================================================

    # Distance to nearest geometry boundary (surface or lattice)
    # Also set particle material and speed
    d_boundary, event = distance_to_boundary(P, mcdc)

    # Distance to tally mesh
    d_mesh = INF
    if mcdc["cycle_active"]:
        for tally in mcdc["mesh_tallies"]:
            d_mesh = min(d_mesh, distance_to_mesh(P, tally["filter"], mcdc))

    d_domain = INF
    if mcdc["cycle_active"] and mcdc["technique"]["domain_decomposition"]:
        d_domain = distance_to_mesh(P, mcdc["technique"]["dd_mesh"], mcdc)

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
    distance = min(
        d_boundary, d_time_boundary, d_time_census, d_mesh, d_collision, d_domain
    )
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
    if mcdc["technique"]["iQMC"]:
        if mcdc["setting"]["track_particle"]:
            track_particle(P, mcdc)
        iqmc_score_tallies(P, distance, mcdc)
        iqmc_continuous_weight_reduction(P, distance, mcdc)
        if abs(P["w"]) <= mcdc["technique"]["iqmc"]["w_min"]:
            P["alive"] = False

    # Score tracklength tallies
    if mcdc["cycle_active"]:
        for tally in mcdc["mesh_tallies"]:
            score_mesh_tally(P, distance, tally, data, mcdc)
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
    trans_struct = adapt.local_translate()
    trans = trans_struct["values"]

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
            for i in range(3):
                P["translation"][i] = trans[i]

            if surface_move:
                event = EVENT_SURFACE_MOVE

        if cell["fill_type"] == FILL_MATERIAL:
            P["material_ID"] = cell["fill_ID"]
            break

        elif cell["fill_type"] == FILL_LATTICE:
            # Get lattice
            lattice = mcdc["lattices"][cell["fill_ID"]]

            # Get lattice center for translation)
            for i in range(3):
                trans[i] -= cell["translation"][i]

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


@njit
def distance_to_tally_mesh(P, mcdc):
    x = P["x"]
    y = P["y"]
    z = P["z"]
    t = P["t"]
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    v = get_particle_speed(P, mcdc)

    d = INF

    for tally in mcdc["mesh_tallies"]:
        d = min(d, mesh_distance_search(x, ux, tally["filter"]["x"]))
        d = min(d, mesh_distance_search(y, uy, tally["filter"]["y"]))
        d = min(d, mesh_distance_search(z, uz, tally["filter"]["z"]))
        d = min(d, mesh_distance_search(t, 1.0 / v, tally["filter"]["t"]))

    return d


# =============================================================================
# Surface crossing
# =============================================================================


@njit
def surface_crossing(P, data, prog):
    mcdc = adapt.device(prog)

    surface = mcdc["surfaces"][P["surface_ID"]]

    # Translation
    trans_struct = adapt.local_translate()
    trans = trans_struct["values"]
    trans = P["translation"]

    # Score tally
    for i in range(surface["N_tally"]):
        ID = surface["tally_IDs"][i]
        tally = mcdc["surface_tallies"][ID]
        score_surface_tally(P, surface, tally, data, mcdc)

    # Implement BC
    surface_bc(P, surface, trans)

    # Small shift to ensure crossing
    surface_shift(P, surface, trans, mcdc)

    # Record old material for sensitivity quantification
    material_ID_old = P["material_ID"]

    # Check new cell?
    if P["alive"] and not surface["BC"] == BC_REFLECTIVE:
        cell = mcdc["cells"][P["cell_ID"]]
        if not cell_check(P, cell, trans, mcdc):
            trans_struct = adapt.local_translate()
            trans = trans_struct["values"]
            P["cell_ID"] = get_particle_cell(P, UNIVERSE_ROOT, trans, mcdc)

    # Sensitivity quantification for surface?
    if surface["sensitivity"] and (
        P["sensitivity_ID"] == 0
        or mcdc["technique"]["dsm_order"] == 2
        and P["sensitivity_ID"] <= mcdc["setting"]["N_sensitivity"]
    ):
        material_ID_new = get_particle_material(P, mcdc)
        if material_ID_old != material_ID_new:
            # Sample derivative source particles
            sensitivity_surface(P, surface, material_ID_old, material_ID_new, prog)


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
def scattering(P, prog):
    mcdc = adapt.device(prog)
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

        # Bank, but keep it if it is the last particle
        if n == N - 1:
            P["alive"] = True
            P["ux"] = P_new["ux"]
            P["uy"] = P_new["uy"]
            P["uz"] = P_new["uz"]
            P["g"] = P_new["g"]
            P["E"] = P_new["E"]
            P["w"] = P_new["w"]
        else:
            adapt.add_active(P_new, prog)


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
def fission(P, prog):
    mcdc = adapt.device(prog)

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
        nu = get_nu(NU_FISSION, nuclide, E)
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
        idx_census = mcdc["idx_census"]
        if P_new["t"] > mcdc["setting"]["census_time"][idx_census]:
            adapt.add_census(P_new, prog)
        elif mcdc["setting"]["mode_eigenvalue"]:
            adapt.add_census(P_new, prog)
        else:
            # Keep it if it is the last particle
            if n == N - 1:
                P["alive"] = True
                P["ux"] = P_new["ux"]
                P["uy"] = P_new["uy"]
                P["uz"] = P_new["uz"]
                P["t"] = P_new["t"]
                P["g"] = P_new["g"]
                P["E"] = P_new["E"]
                P["w"] = P_new["w"]
            else:
                adapt.add_active(P_new, prog)


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
    nu = get_nu(NU_FISSION, nuclide, E)
    nu_p = get_nu(NU_FISSION_PROMPT, nuclide, E)
    nu_d_struct = adapt.local_j_array()
    nu_d = nu_d_struct["values"]
    for j in range(J):
        nu_d[j] = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)

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
def branchless_collision(P, prog):
    mcdc = adapt.device(prog)

    material = mcdc["materials"][P["material_ID"]]

    # Adjust weight
    SigmaT = get_MacroXS(XS_TOTAL, material, P, mcdc)
    n_scatter = get_MacroXS(XS_NU_SCATTER, material, P, mcdc)
    n_fission = get_MacroXS(XS_NU_FISSION, material, P, mcdc) / mcdc["k_eff"]
    n_total = n_fission + n_scatter
    P["w"] *= n_total / SigmaT

    # Set spectrum and decay rate
    if rng(P) < n_scatter / n_total:
        sample_phasespace_scattering(P, material, P, mcdc)
    else:
        if mcdc["setting"]["mode_MG"]:
            sample_phasespace_fission(P, material, P, mcdc)
        else:
            nuclide = sample_nuclide(material, P, XS_NU_FISSION, mcdc)
            sample_phasespace_fission_nuclide(P, nuclide, P, mcdc)

            # Beyond time census or time boundary?
            idx_census = mcdc["idx_census"]
            if P["t"] > mcdc["setting"]["census_time"][idx_census]:
                P["alive"] = False
                adapt.add_active(split_particle(P), prog)
            elif P["t"] > mcdc["setting"]["time_boundary"]:
                P["alive"] = False


# =============================================================================
# Weight widow
# =============================================================================


@njit
def weight_window(P, prog):
    mcdc = adapt.device(prog)

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
            adapt.add_active(split_particle(P), prog)

        # Russian roulette
        p -= n_split
        xi = rng(P)
        if xi <= p:
            adapt.add_active(split_particle(P), prog)

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


@toggle("iQMC")
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
        P_new = adapt.local_particle_record()
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


@toggle("iQMC")
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


@toggle("iQMC")
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


@toggle("iQMC")
def iqmc_sample_position(a, b, sample):
    return a + (b - a) * sample


@toggle("iQMC")
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


@toggle("iQMC")
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
    # variables for cell finding functions
    trans_struct = adapt.local_translate()
    trans = trans_struct["values"]
    # create particle to utilize cell finding functions
    P_temp = adapt.local_particle()
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
                    P_temp["cell_ID"] = get_particle_cell(
                        P_temp, UNIVERSE_ROOT, trans, mcdc
                    )

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


@toggle("iQMC")
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


@toggle("iQMC")
def AxV(V, b, data, mcdc):
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
    set_bank_size(mcdc["bank_source"], 0)

    # QMC Sweep
    iqmc_prepare_particles(mcdc)
    iqmc_reset_tallies(iqmc)
    iqmc["sweep_counter"] += 1
    loop_source(0, data, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_tallies(iqmc)
    # update source adds effective scattering + fission + fixed-source
    iqmc_update_source(mcdc)
    # combine all sources (constant and tilted) into one vector
    iqmc_consolidate_sources(mcdc)
    v_out = iqmc["total_source"].copy()
    axv = V - (v_out - b)

    return axv


# =============================================================================
# Weight Roulette
# =============================================================================


@njit
def weight_roulette(P, mcdc):
    w_survive = mcdc["technique"]["wr_survive"]
    prob_survive = P["w"] / w_survive
    if rng(P) <= prob_survive:
        P["w"] = w_survive
        if mcdc["technique"]["iQMC"]:
            P["iqmc"]["w"][:] = w_survive
    else:
        P["alive"] = False


# ==============================================================================
# Sensitivity quantification (Derivative Source Method)
# =============================================================================


@njit
def sensitivity_surface(P, surface, material_ID_old, material_ID_new, prog):

    mcdc = adapt.device(prog)

    # Sample number of derivative sources
    xi = surface["dsm_Np"]
    if xi != 1.0:
        Np = int(math.floor(xi + rng(P)))
    else:
        Np = 1

    # Terminate and put the current particle into the secondary bank
    P["alive"] = False
    adapt.add_active(copy_record(P), prog)

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
        adapt.add_active(P_new, prog)

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

        for idx in range(2):

            if idx == 0:
                material_ID = material_ID_old
                sign = sign_old
            else:
                material_ID = material_ID_new
                sign = sign_new

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
                            adapt.add_active(P_new, prog)
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
                                adapt.add_active(P_new, prog)
                                source_obtained = True
                            else:
                                tot += nusigmaF
                                if tot > xi:
                                    # Fission source
                                    sample_phasespace_fission_nuclide(
                                        P, nuclide, P_new, mcdc
                                    )
                                    P_new["sensitivity_ID"] = ID_source
                                    adapt.add_active(P_new, prog)
                                    source_obtained = True
                    if source_obtained:
                        break
                if source_obtained:
                    break


@njit
def sensitivity_material(P, prog):

    mcdc = adapt.device(prog)

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
        adapt.add_active(P_new, prog)


# ==============================================================================
# Particle tracker
# ==============================================================================


@toggle("particle_tracker")
def track_particle(P, mcdc):
    idx = adapt.global_add(mcdc["particle_track_N"], 0, 1)
    mcdc["particle_track"][idx, 0] = P["track_hid"]
    mcdc["particle_track"][idx, 1] = P["track_pid"]
    mcdc["particle_track"][idx, 2] = P["g"] + 1
    mcdc["particle_track"][idx, 3] = P["t"]
    mcdc["particle_track"][idx, 4] = P["x"]
    mcdc["particle_track"][idx, 5] = P["y"]
    mcdc["particle_track"][idx, 6] = P["z"]
    mcdc["particle_track"][idx, 7] = P["w"]


@toggle("particle_tracker")
def copy_track_data(P_new, P):
    P_new["track_hid"] = P["track_hid"]
    P_new["track_pid"] = P["track_pid"]


@toggle("particle_tracker")
def allocate_hid(P, mcdc):
    P["track_hid"] = adapt.global_add(mcdc["particle_track_history_ID"], 0, 1)


@toggle("particle_tracker")
def allocate_pid(P, mcdc):
    P["track_pid"] = adapt.global_add(mcdc["particle_track_particle_ID"], 0, 1)


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
        # Cross sections
        if type_ == XS_TOTAL:
            return material["total"][g]
        elif type_ == XS_SCATTER:
            return material["scatter"][g]
        elif type_ == XS_CAPTURE:
            return material["capture"][g]
        elif type_ == XS_FISSION:
            return material["fission"][g]

        # Productions
        elif type_ == XS_NU_SCATTER:
            nu = material["nu_s"][g]
            scatter = material["scatter"][g]
            return nu * scatter
        elif type_ == XS_NU_FISSION:
            nu = material["nu_f"][g]
            fission = material["fission"][g]
            return nu * fission
        elif type_ == XS_NU_FISSION_PROMPT:
            nu_p = material["nu_p"][g]
            fission = material["fission"][g]
            return nu_p * fission
        elif type_ == XS_NU_FISSION_DELAYED:
            nu_d = 0.0
            for j in range(material["J"]):
                nu_d += material["nu_d"][g, j]
            fission = material["fission"][g]
            return nu_d * fission

    # Continuous-energy XS
    MacroXS = 0.0
    E = P["E"]

    # Sum over all nuclides
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]

        # Get nuclide density
        N = material["nuclide_densities"][i]

        # Get microscopic cross-section
        microXS = get_microXS(type_, nuclide, E)

        # Accumulate
        MacroXS += N * microXS

    return MacroXS


@njit
def get_microXS(type_, nuclide, E):
    # Cross sections
    if type_ == XS_TOTAL:
        data = nuclide["ce_total"]
        return get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
    elif type_ == XS_SCATTER:
        data = nuclide["ce_scatter"]
        return get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
    elif type_ == XS_CAPTURE:
        data = nuclide["ce_capture"]
        return get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
    elif type_ == XS_FISSION:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        return get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])

    # Binary Multiplicities
    elif type_ == XS_NU_SCATTER:
        data = nuclide["ce_scatter"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = 1.0
        return nu * xs
    elif type_ == XS_NU_FISSION:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION, nuclide, E)
        return nu * xs
    elif type_ == XS_NU_FISSION_PROMPT:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION_PROMPT, nuclide, E)
        return nu * xs
    elif type_ == XS_NU_FISSION_DELAYED:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION_DELAYED, nuclide, E)
        return nu * xs


@njit
def get_XS(data, E, E_grid, NE):
    # Search XS energy bin index
    idx = binary_search_length(E, E_grid, NE)

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
def get_nu_group(type_, nuclide, E, group):
    if type_ == NU_FISSION:
        nu = get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])
        for i in range(6):
            nu += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return nu

    if type_ == NU_FISSION_PROMPT:
        return get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])

    if type_ == NU_FISSION_DELAYED and group == -1:
        tot = 0.0
        for i in range(6):
            tot += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return tot

    if type_ == NU_FISSION_DELAYED and group != -1:
        return get_XS(
            nuclide["ce_nu_d"][group], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
        )


@njit
def get_nu(type_, nuclide, E):
    return get_nu_group(type_, nuclide, E, -1)


@njit
def sample_nuclide(material, P, type_, mcdc):
    xi = rng(P) * get_MacroXS(type_, material, P, mcdc)
    tot = 0.0
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]

        N = material["nuclide_densities"][i]
        tot += N * get_microXS(type_, nuclide, P["E"])
        if tot > xi:
            break

    return nuclide


@njit
def sample_Eout(P_new, E_grid, NE, chi):
    xi = rng(P_new)

    # Determine bin index
    idx = binary_search_length(xi, chi, NE)

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
def binary_search_length(val, grid, length):
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
def binary_search(val, grid):
    return binary_search_length(val, grid, 0)


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
def uq_tally_closeout_history(data, mcdc):
    tally_bin = data[TALLY]

    # Assumes N_batch > 1
    # Accumulate square of history score, but continue to accumulate bin
    history_bin = tally_bin[TALLY_SCORE] - tally_bin[TALLY_UQ_BATCH]
    tally_bin[TALLY_UQ_BATCH_VAR] += history_bin**2
    tally_bin[TALLY_UQ_BATCH] = tally_bin[TALLY_SCORE]


@njit
def uq_tally_closeout_batch(data, mcdc):
    tally_bin = data[TALLY]

    # Reset bin
    N_bin = tally_bin.shape[1]
    for i in range(N_bin):
        # Reset score bin
        tally_bin[TALLY_UQ_BATCH, i] = 0.0

    # MPI Reduce
    buff = np.zeros(N_bin)
    with objmode():
        MPI.COMM_WORLD.Reduce(np.array(tally_bin[TALLY_UQ_BATCH_VAR]), buff, MPI.SUM, 0)
    tally_bin[TALLY_UQ_BATCH_VAR][:] = buff


@njit
def uq_tally_closeout(data, mcdc):
    tally_bin = data[TALLY]

    N_history = mcdc["setting"]["N_particle"]

    tally_bin[TALLY_UQ_BATCH_VAR] = (
        tally_bin[TALLY_UQ_BATCH_VAR] / N_history - tally_bin[TALLY_SUM_SQ]
    ) / (N_history - 1)

    # If we're here, N_batch > 1
    N_history = mcdc["setting"]["N_batch"]

    # Store results
    mean = tally_bin[TALLY_SUM] / N_history
    tally_bin[TALLY_UQ_BATCH_VAR] /= N_history
    tally_bin[TALLY_UQ_BATCH] = (
        tally_bin[TALLY_SUM_SQ] - N_history * np.square(mean)
    ) / (N_history - 1)
