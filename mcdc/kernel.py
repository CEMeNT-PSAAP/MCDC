import math

from mpi4py import MPI
from numba import njit, objmode, literal_unroll
import numba

import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import print_error
from mcdc.type_ import score_list
from mcdc.loop import loop_source


# =============================================================================
# Domain Decomposition
# =============================================================================


# =============================================================================
# Domain crossing event
# =============================================================================


@njit
def domain_crossing(P, mcdc):
    # Domain mesh crossing
    max_size = mcdc["technique"]["exchange_rate"]  # /1e4)*mcdc["mpi_work_size"])
    if mcdc["technique"]["domain_decomp"] and P["alive"] == True:
        mesh = mcdc["technique"]["domain_mesh"]
        # Determine which dimension is crossed
        x, y, z, t, flag = mesh_crossing_evaluate(P, mesh)
        # Score on tally
        if flag == MESH_X and P["ux"] > 0:
            add_particle(copy_particle(P), mcdc["bank_domain_xp"])
            if mcdc["bank_domain_xp"]["size"] == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_X and P["ux"] < 0:
            add_particle(copy_particle(P), mcdc["bank_domain_xn"])
            if mcdc["bank_domain_xn"]["size"] == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Y and P["uy"] > 0:
            add_particle(copy_particle(P), mcdc["bank_domain_yp"])
            if mcdc["bank_domain_yp"]["size"] == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Y and P["uy"] < 0:
            add_particle(copy_particle(P), mcdc["bank_domain_yn"])
            if mcdc["bank_domain_yn"]["size"] == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Z and P["uz"] > 0:
            add_particle(copy_particle(P), mcdc["bank_domain_zp"])
            if mcdc["bank_domain_zp"]["size"] == max_size:
                dd_particle_send(mcdc)
        if flag == MESH_Z and P["uz"] < 0:
            add_particle(copy_particle(P), mcdc["bank_domain_zn"])
            if mcdc["bank_domain_zn"]["size"] == max_size:
                dd_particle_send(mcdc)
        P["alive"] = False


# =============================================================================
# Send full domain bank
# =============================================================================


def dd_particle_send(mcdc):
    with objmode(size="int64"):
        for i in range(
            max(
                len(mcdc["technique"]["xp_neigh"]),
                len(mcdc["technique"]["xn_neigh"]),
                len(mcdc["technique"]["yp_neigh"]),
                len(mcdc["technique"]["yn_neigh"]),
                len(mcdc["technique"]["zp_neigh"]),
                len(mcdc["technique"]["zn_neigh"]),
            )
        ):
            if mcdc["technique"]["xp_neigh"].size > i:
                size = mcdc["bank_domain_xp"]["size"]
                ratio = int(size/len(mcdc["technique"]["xp_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["xp_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_xp"]["particles"][start:end])
                request1 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["xp_neigh"][i], tag=1
                )
    

            if mcdc["technique"]["xn_neigh"].size > i:
                size = mcdc["bank_domain_xn"]["size"]
                ratio = int(size/len(mcdc["technique"]["xn_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["xn_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_xn"]["particles"][start:end])
                request2 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["xn_neigh"][i], tag=2
                )

            if mcdc["technique"]["yp_neigh"].size > i:
                size = mcdc["bank_domain_yp"]["size"]
                ratio = int(size/len(mcdc["technique"]["yp_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["yp_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_yp"]["particles"][start:end])
                request3 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["yp_neigh"][i], tag=3
                )

            if mcdc["technique"]["yn_neigh"].size > i:
                size = mcdc["bank_domain_yn"]["size"]
                ratio = int(size/len(mcdc["technique"]["yn_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["yn_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_yn"]["particles"][start:end])
                request4 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["yn_neigh"][i], tag=4
                )

            if mcdc["technique"]["zp_neigh"].size > i:
                size = mcdc["bank_domain_zp"]["size"]
                ratio = int(size/len(mcdc["technique"]["zp_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["zp_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_zp"]["particles"][start:end])
                request5 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["zp_neigh"][i], tag=5
                )

            if mcdc["technique"]["zn_neigh"].size > i:
                size = mcdc["bank_domain_zn"]["size"]
                ratio = int(size/len(mcdc["technique"]["zn_neigh"]))
                start = ratio*i
                end =start+ratio
                if i == len(mcdc["technique"]["zn_neigh"])-1:
                    end = size
                bank = np.array(mcdc["bank_domain_zn"]["particles"][start:end])
                request6 = MPI.COMM_WORLD.send(
                    bank, dest=mcdc["technique"]["zn_neigh"][i], tag=6
                )

        mcdc["bank_domain_xp"]["size"] = 0
        mcdc["bank_domain_xn"]["size"] = 0
        mcdc["bank_domain_yp"]["size"] = 0
        mcdc["bank_domain_yn"]["size"] = 0
        mcdc["bank_domain_zp"]["size"] = 0
        mcdc["bank_domain_zn"]["size"] = 0


# =============================================================================
# Recieve particles and clear banks
# =============================================================================


def dd_particle_receive(mcdc):
    buff = np.zeros(
        mcdc["bank_domain_xp"]["particles"].shape[0], dtype=type_.particle_record
    )

    with objmode(size="int64"):
        bankr = mcdc["bank_active"]["particles"][:0]
        size_old = bankr.shape[0]
        for i in range(
            max(
                len(mcdc["technique"]["xp_neigh"]),
                len(mcdc["technique"]["xn_neigh"]),
                len(mcdc["technique"]["yp_neigh"]),
                len(mcdc["technique"]["yn_neigh"]),
                len(mcdc["technique"]["zp_neigh"]),
                len(mcdc["technique"]["zn_neigh"]),
            )
        ):
            if mcdc["technique"]["xp_neigh"].size > i:
                received1 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["xp_neigh"][i], tag=2
                )
                if received1.Get_status():
                    bankr = np.append(bankr, received1.wait())
                else:
                    MPI.Request.cancel(received1)
            if mcdc["technique"]["xn_neigh"].size > i:
                received2 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["xn_neigh"][i], tag=1
                )
                if received2.Get_status():
                    bankr = np.append(bankr, received2.wait())
                else:
                    MPI.Request.cancel(received2)
            if mcdc["technique"]["yp_neigh"].size > i:
                received3 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["yp_neigh"][i], tag=4
                )
                if received3.Get_status():
                    bankr = np.append(bankr, received3.wait())
                else:
                    MPI.Request.cancel(received3)
            if mcdc["technique"]["yn_neigh"].size > i:
                received4 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["yn_neigh"][i], tag=3
                )
                if received4.Get_status():
                    bankr = np.append(bankr, received4.wait())
                else:
                    MPI.Request.cancel(received4)
            if mcdc["technique"]["zp_neigh"].size > i:
                received5 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["zp_neigh"][i], tag=6
                )
                if received5.Get_status():
                    bankr = np.append(bankr, received5.wait())
                else:
                    MPI.Request.cancel(received5)

            if mcdc["technique"]["zn_neigh"].size > i:
                received6 = MPI.COMM_WORLD.irecv(
                    source=mcdc["technique"]["zn_neigh"][i], tag=5
                )
                if received6.Get_status():
                    bankr = np.append(bankr, received6.wait())
                else:
                    MPI.Request.cancel(received6)

        size = bankr.shape[0]
        # Set output buffer
        for i in range(size):
            buff[i] = bankr[i]
        # if (size-size_old)>0:
        #print("recieved",size-size_old,"particles, in domain",mcdc["d_idx"])
    # Set source bank from buffer
    for i in range(size):
        add_particle(buff[i], mcdc["bank_active"])


# =============================================================================
# Particle in domain
# =============================================================================


# Check if particle is in domain
def particle_in_domain(P, mcdc):
    d_idx = mcdc["d_idx"]
    d_Nx = mcdc["technique"]["domain_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["domain_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["domain_mesh"]["z"].size - 1

    d_iz=int(d_idx/(d_Nx*d_Ny))
    d_iy=int((d_idx-d_Nx*d_Ny*d_iz)/d_Nx)
    d_ix=int(d_idx-d_Nx*d_Ny*d_iz-d_Nx*d_iy)

    x_cell = binary_search(P["x"], mcdc["technique"]["domain_mesh"]["x"])
    y_cell = binary_search(P["y"], mcdc["technique"]["domain_mesh"]["y"])
    z_cell = binary_search(P["z"], mcdc["technique"]["domain_mesh"]["z"])
    # print("xc",x_cell,"yc",y_cell,"zc",z_cell,"x",P["x"],"y",P["y"],"z",P["z"],"d_x",d_ix,"d_y",d_iy,"d_z",d_iz,"d_idx",d_idx)
    if d_ix == x_cell:
        if d_iy == y_cell:
            if d_iz == z_cell:
                return True
    else:
        return False


# =============================================================================
# Source in domain
# =============================================================================


# Check for source in domain
def source_in_domain(source, domain_mesh, d_idx):
    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz=int(d_idx/(d_Nx*d_Ny))
    d_iy=int((d_idx-d_Nx*d_Ny*d_iz)/d_Nx)
    d_ix=int(d_idx-d_Nx*d_Ny*d_iz-d_Nx*d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    # print("domain:",d_idx,"d_x:",d_x,"d_y:",d_y,"d_z:",d_z,domain_mesh["z"],source["box_x"],source["box_y"],source["box_z"])

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


def domain_work(mcdc, domain, N):
    domain_mesh = mcdc["technique"]["domain_mesh"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1
    work_start = 0
    for d_idx in range(domain):
        d_iz=int(d_idx/(d_Nx*d_Ny))
        d_iy=int((d_idx-d_Nx*d_Ny*d_iz)/d_Nx)
        d_ix=int(d_idx-d_Nx*d_Ny*d_iz-d_Nx*d_iy)

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
    d_iz=int(mcdc["d_idx"]/(d_Nx*d_Ny))
    d_iy=int((mcdc["d_idx"]-d_Nx*d_Ny*d_iz)/d_Nx)
    d_ix=int(mcdc["d_idx"]-d_Nx*d_Ny*d_iz-d_Nx*d_iy)

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

        xV = source["box_x"][1] - source["box_x"][0]
        if xV != 0:
            Vi[i] *= xV
            Vim[i] *= min(source["box_x"][1], d_x[1]) - max(source["box_x"][0], d_x[0])
        yV = source["box_y"][1] - source["box_y"][0]
        if yV != 0:
            Vi[i] *= yV
            Vim[i] *= min(source["box_y"][1], d_y[1]) - max(source["box_y"][0], d_y[0])
        zV = source["box_z"][1] - source["box_z"][0]
        if zV != 0:
            Vi[i] *= zV
            Vim[i] *= min(source["box_z"][1], d_z[1]) - max(source["box_z"][0], d_z[0])
        if not source_in_domain(source, domain_mesh, d_idx):
            Vim[i] = 0
        i += 1
    for source in range(num_source):
        Nm += Ni[source] * Vim[source] / Vi[source]
    Nm /= mcdc["technique"]["work_ratio"][domain]
    rank = MPI.COMM_WORLD.Get_rank()
    if mcdc["technique"]["work_ratio"][domain] > 1:
        work_start += Nm * (rank - np.sum(mcdc["technique"]["work_ratio"][0:d_idx]))
    print("domain:", d_idx, ", Nm:", int(Nm), ", work_start:", int(work_start))
    total_v=0
    for source in range(len(mcdc["sources"])):
        total_v+=Vim[source]
    i = 0
    for source in mcdc["sources"]:
        source["prob"]*=2*Vim[i]/total_v
        print("domain:", d_idx,Vim[i]/total_v,"source",source["prob"],i)
        i+=1
    return (int(Nm), int(work_start))


# =============================================================================
# Source particle in domain only
# =============================================================================


@njit
def source_particle_dd(seed, mcdc):
    domain_mesh = mcdc["technique"]["domain_mesh"]
    d_idx = mcdc["d_idx"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz=int(mcdc["d_idx"]/(d_Nx*d_Ny))
    d_iy=int((mcdc["d_idx"]-d_Nx*d_Ny*d_iz)/d_Nx)
    d_ix=int(mcdc["d_idx"]-d_Nx*d_Ny*d_iz-d_Nx*d_iy)


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
    P["w"] = 1  # /(mcdc["technique"]["work_ratio"][mcdc["d_idx"]])
    # P["w"] =np.sum(mcdc["technique"]["work_ratio"])/(mcdc["technique"]["work_ratio"][d_idx])#len(mcdc["sources"])*(1+(np.sum(mcdc["technique"]["work_ratio"])-len(mcdc["technique"]["work_ratio"]))/len(mcdc["technique"]["work_ratio"]))/(mcdc["technique"]["work_ratio"][d_idx])

    P["sensitivity_ID"] = 0
    return P


@njit
def distribute_work_dd(N, mcdc, precursor=False):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    source_alloc = int(
        N * len(mcdc["sources"]) / (len(mcdc["technique"]["work_ratio"]))
    )
    print("source-alloc", source_alloc)
    # Total # of work
    work_size_total = N
    if rank == 0:
        # Evenly distribute work
        work_size = 0
        for source in mcdc["sources"]:
            if source_in_domain(
                source, mcdc["technique"]["domain_mesh"], mcdc["d_idx"]
            ):
                work_size += source_alloc
        work_start = 0
    else:
        work_start = MPI.COMM_WORLD.recv(source=rank - 1, tag=0)
        print("WORK_START", work_start)
        work_size = 0
        for source in mcdc["sources"]:
            if source_in_domain(
                source, mcdc["technique"]["domain_mesh"], mcdc["d_idx"]
            ):
                work_size += int(source_alloc * source["prob"])

    work_finish = work_size + work_start
    if rank < size - 1:
        send = MPI.COMM_WORLD.send(work_finish, dest=rank + 1, tag=0)
    work_size, work_start = domain_work(mcdc, mcdc["d_idx"], N)

            

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total

    print(
        "Rank",
        rank,
        "work_size:",
        work_size,
        "work_start:",
        work_start,
        "work_size_total:",
        work_size_total,
    )


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
    P["w"] = P_rec["w"]
    P["rng_seed"] = P_rec["rng_seed"]

    if mcdc["technique"]["iQMC"]:
        P["iqmc_w"] = P_rec["iqmc_w"]

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
    return mcdc["materials"][P["material_ID"]]["speed"][P["g"]]


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
def mesh_get_energy_index(P, mesh):
    return binary_search(P["g"], mesh["g"])


@njit
def mesh_uniform_get_index(P, mesh, trans):
    Px = P["x"] + trans[0]
    Py = P["y"] + trans[1]
    Pz = P["z"] + trans[2]
    x = math.floor((Px - mesh["x0"]) / mesh["dx"])
    y = math.floor((Py - mesh["y0"]) / mesh["dy"])
    z = math.floor((Pz - mesh["z0"]) / mesh["dz"])
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
    if x1 != x2:
        return x1, y1, z1, t1, MESH_X
    elif y1 != y2:
        return x1, y1, z1, t1, MESH_Y
    elif z1 != z2:
        return x1, y1, z1, t1, MESH_Z
    elif t1 != t2:
        return x1, y1, z1, t1, MESH_T


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
    g = mesh_get_energy_index(P, tally["mesh"])

    # Outside grid?
    if outside:
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

    d_domain = INF
    if mcdc["cycle_active"] and mcdc["technique"]["domain_decomp"]:
        d_domain = distance_to_mesh(P, mcdc["technique"]["domain_mesh"], mcdc)

    if mcdc["technique"]["iQMC"]:
        d_iqmc_mesh = distance_to_mesh(P, mcdc["technique"]["iqmc_mesh"], mcdc)
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
        material = mcdc["materials"][P["material_ID"]]
        w = P["iqmc_w"]
        SigmaT = material["total"][:]
        score_iqmc_flux(P, distance, mcdc)
        w_final = continuous_weight_reduction(w, distance, SigmaT)
        P["iqmc_w"] = w_final
        P["w"] = w_final.sum()

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
    SigmaT = material["total"][P["g"]]

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
    if surface["vacuum"]:
        mcdc["p_comp"]+=1
    # Small shift to ensure crossing
    surface_shift(P, surface, trans, mcdc)

    # Record old material for sensitivity quantification
    material_ID_old = P["material_ID"]

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
    SigmaT = material["total"][g]
    SigmaC = material["capture"][g]
    SigmaS = material["scatter"][g]
    SigmaF = material["fission"][g]

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
        mcdc["p_comp"]+=1
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

    # Get production factor
    material = mcdc["materials"][P["material_ID"]]
    g = P["g"]
    nu_s = material["nu_s"][g]

    # Get number of secondaries
    N = int(math.floor(weight_eff * nu_s + rng(P)))

    for n in range(N):
        # Create new particle
        P_new = split_particle(P)

        # Set weight
        P_new["w"] = weight_new

        # Sample scattering phase space
        sample_phasespace_scattering(P, material, P_new)

        # Bank
        add_particle(P_new, mcdc["bank_active"])


@njit
def sample_phasespace_scattering(P, material, P_new):
    # Get outgoing spectrum
    g = P["g"]
    G = material["G"]
    chi_s = material["chi_s"][g]

    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["sensitivity_ID"] = P["sensitivity_ID"]

    # Sample outgoing energy
    xi = rng(P_new)
    tot = 0.0
    for g_out in range(G):
        tot += chi_s[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out

    # Sample scattering angle
    mu = 2.0 * rng(P_new) - 1.0

    # Sample azimuthal direction
    azi = 2.0 * PI * rng(P_new)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu**2) ** 0.5

    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    if uz != 1.0:
        B = (1.0 - P["uz"] ** 2) ** 0.5
        C = Ac / B

        P_new["ux"] = ux * mu + (ux * uz * cos_azi - uy * sin_azi) * C
        P_new["uy"] = uy * mu + (uy * uz * cos_azi + ux * sin_azi) * C
        P_new["uz"] = uz * mu - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2) ** 0.5
        C = Ac / B

        P_new["ux"] = ux * mu + (ux * uy * cos_azi - uz * sin_azi) * C
        P_new["uz"] = uz * mu + (uz * uy * cos_azi + ux * sin_azi) * C
        P_new["uy"] = uy * mu - cos_azi * Ac * B


# =============================================================================
# Fission
# =============================================================================


@njit
def fission(P, mcdc):
    # Kill the current particle
    P["alive"] = False

    # Get production factor
    material = mcdc["materials"][P["material_ID"]]
    g = P["g"]
    nu = material["nu_f"][g]

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Get number of secondaries
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + rng(P)))

    for n in range(N):
        # Create new particle
        P_new = split_particle(P)

        # Set weight
        P_new["w"] = weight_new

        # Sample scattering phase space
        sample_phasespace_fission(P, material, P_new, mcdc)

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
                SigmaF = material["fission"][g]
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
    # Get constants
    G = nuclide["G"]
    J = nuclide["J"]
    g = P["g"]
    nu = nuclide["nu_f"][g]
    nu_p = nuclide["nu_p"][g]
    if J > 0:
        nu_d = nuclide["nu_d"][g]

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
    SigmaF = material["fission"][g]
    SigmaS = material["scatter"][g]
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
def continuous_weight_reduction(w, distance, SigmaT):
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
    return w * np.exp(-distance * SigmaT)


@njit
def UpdateK(keff, phi_outter, phi_inner, mcdc):
    mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    nuSigmaF = np.zeros_like(phi_outter)
    # calculate nu*SigmaF for every cell
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t = 0
                mat_idx = mcdc["technique"]["iqmc_material_idx"][t, i, j, k]
                material = mcdc["materials"][mat_idx]
                nu_f = material["nu_f"]
                SigmaF = material["fission"]
                nuSigmaF[:, t, i, j, k] = nu_f * SigmaF

    keff *= np.sum(nuSigmaF * phi_inner) / np.sum(nuSigmaF * phi_outter)
    mcdc["k_eff"] = keff


@njit
def prepare_qmc_source(mcdc):
    """

    Iterates trhough all spatial cells to calculate the iQMC source. The source
    is a combination of the user input Fixed-Source plus the calculated
    Scattering-Source and Fission-Sources. Resutls are stored in
    mcdc['technique']['iqmc_source'], a matrix of size [G,Nt,Nx,Ny,Nz].

    """
    Q = mcdc["technique"]["iqmc_source"]
    fixed_source = mcdc["technique"]["iqmc_fixed_source"]
    flux_scatter = mcdc["technique"]["iqmc_flux"]
    flux_fission = mcdc["technique"]["iqmc_flux"]
    if mcdc["setting"]["mode_eigenvalue"]:
        flux_fission = mcdc["technique"]["iqmc_flux_outter"]
    mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # calculate source for every cell and group in the iqmc_mesh
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t = 0
                mat_idx = mcdc["technique"]["iqmc_material_idx"][t, i, j, k]
                # we can vectorize the multigroup calculation here
                Q[:, t, i, j, k] = (
                    fission_source(flux_fission[:, t, i, j, k], mat_idx, mcdc)
                    + scattering_source(flux_scatter[:, t, i, j, k], mat_idx, mcdc)
                    + fixed_source[:, t, i, j, k]
                )


@njit
def prepare_qmc_scattering_source(mcdc):
    """

    Iterates trhough all spatial cells to calculate the iQMC scattering-source.
    Resutls are stored in mcdc['technique']['iqmc_source'], a matrix
    of size [G,Nt,Nx,Ny,Nz].

    """
    Q = mcdc["technique"]["iqmc_source"]
    fixed_source = mcdc["technique"]["iqmc_fixed_source"]
    flux = mcdc["technique"]["iqmc_flux"]
    mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # calculate source for every cell and group in the iqmc_mesh
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t = 0
                mat_idx = mcdc["technique"]["iqmc_material_idx"][t, i, j, k]
                # we can vectorize the multigroup calculation here
                Q[:, t, i, j, k] = (
                    scattering_source(flux[:, t, i, j, k], mat_idx, mcdc)
                    + fixed_source[:, t, i, j, k]
                )


@njit
def prepare_qmc_fission_source(mcdc):
    """

    Iterates trhough all spatial cells to calculate the iQMC fission-source.
    Resutls are stored in mcdc['technique']['iqmc_source'], a matrix
    of size [G,Nt,Nx,Ny,Nz].

    """
    Q = mcdc["technique"]["iqmc_source"]
    fixed_source = mcdc["technique"]["iqmc_fixed_source"]
    flux = mcdc["technique"]["iqmc_flux"]
    mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # calculate source for every cell and group in the iqmc_mesh
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t = 0
                mat_idx = mcdc["technique"]["iqmc_material_idx"][t, i, j, k]
                # we can vectorize the multigroup calculation here
                Q[:, t, i, j, k] = (
                    fission_source(flux[:, t, i, j, k], mat_idx, mcdc)
                    + fixed_source[:, t, i, j, k]
                )


@njit
def prepare_qmc_particles(mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    """
    # total number of particles
    N_particle = mcdc["setting"]["N_particle"]
    # number of particles this processor will handle
    N_work = mcdc["mpi_work_size"]

    # low discrepency sequence
    lds = mcdc["technique"]["iqmc_lds"]
    # source
    Q = mcdc["technique"]["iqmc_source"]
    mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # total number of spatial cells
    Nt = Nx * Ny * Nz
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
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = sample_qmc_position(xa, xb, lds[n, 0])
        P_new["y"] = sample_qmc_position(ya, yb, lds[n, 4])
        P_new["z"] = sample_qmc_position(za, zb, lds[n, 3])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = sample_qmc_isotropic_direction(
            lds[n, 1], lds[n, 5]
        )
        if P_new["ux"] == 0.0:
            P_new["ux"] += 0.01
        # time and group
        P_new["t"] = 0
        P_new["g"] = 0
        t, x, y, z, outside = mesh_get_index(P_new, mesh)
        mat_idx = mcdc["technique"]["iqmc_material_idx"][t, x, y, z]
        # calculate dx,dy,dz and then dV
        # TODO: Bug where if x = 0.0 the x-index is -1
        dV = iqmc_cell_volume(x, y, z, mesh)
        # Set weight
        P_new["iqmc_w"] = Q[:, t, x, y, z] * dV * Nt / N_particle
        P_new["w"] = (P_new["iqmc_w"]).sum()
        # add to source bank
        add_particle(P_new, mcdc["bank_source"])


@njit
def fission_source(phi, mat_idx, mcdc):
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
    material = mcdc["nuclides"][mat_idx]
    chi_p = material["chi_p"]
    chi_d = material["chi_d"]
    nu_p = material["nu_p"]
    nu_d = material["nu_d"]
    J = material["J"]
    if mcdc["technique"]["iqmc_eigenmode_solver"] == "davidson":
        keff = 1.0
    else:
        keff = mcdc["k_eff"]

    SigmaF = material["fission"]
    F_p = np.dot(chi_p.T, nu_p / keff * SigmaF * phi)
    F_d = np.dot(chi_d.T, (nu_d.T / keff * SigmaF * phi).sum(axis=1))
    F = F_p + F_d

    return F


@njit
def scattering_source(phi, mat_idx, mcdc):
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
    material = mcdc["materials"][mat_idx]
    chi_s = material["chi_s"]
    SigmaS = material["scatter"]
    return np.dot(chi_s.T, SigmaS * phi)


@njit
def qmc_res(flux_new, flux_old):
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
    flux_new = flux_new.reshape((size,))
    flux_old = flux_old.reshape((size,))
    res = np.linalg.norm((flux_new - flux_old), ord=2)
    return res


@njit
def score_iqmc_flux(P, distance, mcdc):
    """

    Tally the scalar flux and effective fission/scattering rates.

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
    # Get indices
    mesh = mcdc["technique"]["iqmc_mesh"]
    material = mcdc["materials"][P["material_ID"]]
    w = P["iqmc_w"]
    SigmaT = material["total"]
    SigmaS = material["scatter"]
    SigmaF = material["fission"]
    t, x, y, z, outside = mesh_get_index(P, mesh)
    # Outside grid?
    if outside:
        return
    dV = iqmc_cell_volume(x, y, z, mesh)
    # Score
    if SigmaT.all() > 0.0:
        flux = w * (1 - np.exp(-(distance * SigmaT))) / (SigmaT * dV)
    else:
        flux = distance * w / dV
    mcdc["technique"]["iqmc_flux"][:, t, x, y, z] += flux


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
def sample_qmc_position(a, b, sample):
    return a + (b - a) * sample


@njit
def sample_qmc_isotropic_direction(sample1, sample2):
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
def sample_qmc_group(sample, G):
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
def generate_iqmc_material_idx(mcdc):
    """
    This algorithm is meant to loop through every spatial cell of the
    iQMC mesh and assign a material index according to the material_ID at
    the center of the cell.

    Therefore, the whole cell is treated as the material located at the
    center of the cell, regardless of whethere there are more materials
    present.

    A somewhat crude but effient approximation.
    """
    iqmc_mesh = mcdc["technique"]["iqmc_mesh"]
    Nx = len(iqmc_mesh["x"]) - 1
    Ny = len(iqmc_mesh["y"]) - 1
    Nz = len(iqmc_mesh["z"]) - 1
    dx = dy = dz = 1
    t = 0
    # variables for cell finding functions
    trans = np.zeros((3,))
    # create particle to utilize cell finding functions
    P_temp = np.zeros(1, dtype=type_.particle)[0]
    # set default attributes
    P_temp["alive"] = True
    P_temp["material_ID"] = -1
    P_temp["cell_ID"] = -1

    x_mid = 0.5 * (iqmc_mesh["x"][1:] + iqmc_mesh["x"][:-1])
    y_mid = 0.5 * (iqmc_mesh["y"][1:] + iqmc_mesh["y"][:-1])
    z_mid = 0.5 * (iqmc_mesh["z"][1:] + iqmc_mesh["z"][:-1])

    # loop through every cell
    for i in range(Nx):
        x = x_mid[i]
        for j in range(Ny):
            y = y_mid[j]
            for k in range(Nz):
                z = z_mid[k]

                # assign cell center position
                P_temp["x"] = x
                P_temp["y"] = y
                P_temp["z"] = z

                # set cell_ID
                P_temp["cell_ID"] = get_particle_cell(P_temp, 0, trans, mcdc)

                # set material_ID
                material_ID = get_particle_material(P_temp, mcdc)

                # assign material index
                mcdc["technique"]["iqmc_material_idx"][t, i, j, k] = material_ID


@njit
def iqmc_distribute_flux(mcdc):
    flux_local = mcdc["technique"]["iqmc_flux"].copy()
    # TODO: is there a way to do this without creating a new matrix ?
    flux_total = np.zeros_like(flux_local, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(flux_local, flux_total, op=MPI.SUM)
    mcdc["technique"]["iqmc_flux"] = flux_total.copy()


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
    V = np.ascontiguousarray(V)
    w1 = u - np.dot(V, np.dot(V.T, u))
    v1 = w1 / np.linalg.norm(w1)
    w2 = v1 - np.dot(V, np.dot(V.T, v1))
    v2 = w2 / np.linalg.norm(w2)
    V = np.append(V, v2, axis=1)
    # TODO: unit test that each column of V.dot(v2) == 0
    # for i in range(V.shape[1]):
    #     temp = V[:,i].dot(v2)
    #     try:
    #         assert np.isclose(temp, 0.0)
    #     except:
    #         print(temp)
    return V


# =============================================================================
# iQMC Iterative Mapping Functions
# =============================================================================


@njit
def AxV(phi, b, mcdc):
    """
    Linear operator to be used with GMRES.
    """
    matrix_shape = mcdc["technique"]["iqmc_flux"].shape
    vector_size = mcdc["technique"]["iqmc_flux"].size

    mcdc["technique"]["iqmc_flux"] = np.reshape(phi, matrix_shape)

    # reset bank size
    mcdc["bank_source"]["size"] = 0
    mcdc["technique"]["iqmc_source"] = np.zeros_like(mcdc["technique"]["iqmc_source"])

    # QMC Sweep
    prepare_qmc_source(mcdc)
    prepare_qmc_particles(mcdc)
    mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])
    mcdc["technique"]["iqmc_sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_flux(mcdc)

    v_out = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), (vector_size,))
    axv = phi - (v_out - b)

    return axv


@njit
def RHS(mcdc):
    """
    We solve A x = b with a Krylov method. This function extracts
    b by doing a transport sweep of the fixed-source.
    """
    # reshape v and assign to iqmc_flux
    Nt = mcdc["technique"]["iqmc_flux"].size
    mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])

    # reset bank size
    mcdc["bank_source"]["size"] = 0
    mcdc["technique"]["iqmc_source"] = np.zeros_like(mcdc["technique"]["iqmc_source"])

    # QMC Sweep
    prepare_qmc_source(mcdc)
    prepare_qmc_particles(mcdc)
    mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])
    mcdc["technique"]["iqmc_sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_flux(mcdc)

    b = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), (Nt,))

    return b


@njit
def HxV(V, mcdc):
    """
    Linear operator for Davidson method,
    scattering + streaming terms -> (I-L^(-1)S)*phi
    """
    # flux input is most recent iteration of eigenvector
    v = V[:, -1]
    # reshape v and assign to iqmc_flux
    vector_size = v.size
    matrix_shape = mcdc["technique"]["iqmc_flux"].shape
    mcdc["technique"]["iqmc_flux"] = np.reshape(v.copy(), matrix_shape)
    # reset bank size
    mcdc["bank_source"]["size"] = 0
    mcdc["technique"]["iqmc_source"] = np.zeros_like(mcdc["technique"]["iqmc_source"])

    # QMC Sweep
    prepare_qmc_scattering_source(mcdc)
    prepare_qmc_particles(mcdc)
    mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])
    mcdc["technique"]["iqmc_sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_flux(mcdc)

    v_out = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), (vector_size,))
    axv = v - v_out
    axv = np.reshape(axv, (vector_size, 1))

    return axv


@njit
def FxV(V, mcdc):
    """
    Linear operator for Davidson method,
    fission term -> (L^(-1)F*phi)
    """
    # flux input is most recent iteration of eigenvector
    v = V[:, -1]
    # reshape v and assign to iqmc_flux
    vector_size = v.size
    matrix_shape = mcdc["technique"]["iqmc_flux"].shape
    mcdc["technique"]["iqmc_flux"] = np.reshape(v.copy(), matrix_shape)

    # reset bank size
    mcdc["bank_source"]["size"] = 0
    mcdc["technique"]["iqmc_source"] = np.zeros_like(mcdc["technique"]["iqmc_source"])

    # QMC Sweep
    prepare_qmc_fission_source(mcdc)
    prepare_qmc_particles(mcdc)
    mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])
    mcdc["technique"]["iqmc_sweep_counter"] += 1
    loop_source(0, mcdc)
    # sum resultant flux on all processors
    iqmc_distribute_flux(mcdc)

    v_out = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), (vector_size, 1))

    return v_out


@njit
def preconditioner(V, mcdc, num_sweeps=3):
    """
    Linear operator approximation of (I-L^(-1)S)*phi

    In this case the preconditioner is a specified number of purely scattering
    transport sweeps.
    """
    # flux input is most recent iteration of eigenvector
    v = V[:, -1]
    # reshape v and assign to iqmc_flux
    vector_size = v.size
    matrix_shape = mcdc["technique"]["iqmc_flux"].shape
    mcdc["technique"]["iqmc_flux"] = np.reshape(v.copy(), matrix_shape)

    for i in range(num_sweeps):
        # reset bank size
        mcdc["bank_source"]["size"] = 0
        mcdc["technique"]["iqmc_source"] = np.zeros_like(
            mcdc["technique"]["iqmc_source"]
        )

        # QMC Sweep
        prepare_qmc_scattering_source(mcdc)
        prepare_qmc_particles(mcdc)
        mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])
        mcdc["technique"]["iqmc_sweep_counter"] += 1
        loop_source(0, mcdc)
        # sum resultant flux on all processors
        iqmc_distribute_flux(mcdc)

    v_out = np.reshape(mcdc["technique"]["iqmc_flux"].copy(), (vector_size,))
    v_out = v - v_out
    v_out = np.reshape(v_out, (vector_size, 1))

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
        P["iqmc_w"] /= chance
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
                    sample_phasespace_scattering(P, material_old, P_new)
                    P_new["w"] = w_s * sign_old
                else:
                    sample_phasespace_scattering(P, material_new, P_new)
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
                                sample_phasespace_scattering(P, nuclide, P_new)
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
                sample_phasespace_scattering(P, nuclide, P_new)
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
# Miscellany
# =============================================================================


@njit
def binary_search(val, grid):
    """
    Binary search that returns the bin index of a value val given grid grid

    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                                 (-1 if val = min(grid)
    """

    left = 0
    right = len(grid) - 1
    mid = -1
    while left <= right:
        mid = int((left + right) / 2)
        if grid[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return int(right)
