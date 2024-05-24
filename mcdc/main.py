import argparse, os, sys
import numba as nb

# Parse command-line arguments
#   TODO: Will be inside run() once Python/Numba adapter is integrated
parser = argparse.ArgumentParser(description="MC/DC: Monte Carlo Dynamic Code")
parser.add_argument(
    "--mode",
    type=str,
    help="Run mode",
    choices=["python", "numba", "numba_debug"],
    default="python",
)

parser.add_argument(
    "--target", type=str, help="Target", choices=["cpu", "gpu"], default="cpu"
)

parser.add_argument("--N_particle", type=int, help="Number of particles")
parser.add_argument("--output", type=str, help="Output file name")
parser.add_argument("--progress_bar", default=True, action="store_true")
parser.add_argument("--no-progress_bar", dest="progress_bar", action="store_false")
args, unargs = parser.parse_known_args()

from mcdc.print_ import (
    print_banner,
    print_msg,
    print_runtime,
    print_header_eigenvalue,
    print_warning,
    print_error,
)

# Set mode
#   TODO: Will be inside run() once Python/Numba adapter is integrated
mode = args.mode
target = args.target
if mode == "python":
    nb.config.DISABLE_JIT = True
elif mode == "numba":
    nb.config.DISABLE_JIT = False
    nb.config.NUMBA_DEBUG_CACHE = 1
elif mode == "numba_debug":
    msg = "\n >> Entering numba debug mode\n >> will result in slower code and longer compile times\n >> to configure debug options see main.py"
    print_warning(msg)

    nb.config.DISABLE_JIT = False  # turns on the jitter
    nb.config.DEBUG = False  # turns on debugging options
    nb.config.NUMBA_FULL_TRACEBACKS = (
        1  # enables errors from sub-packages to be printed
    )
    nb.config.NUMBA_BOUNDSCHECK = 1  # checks bounds errors of vectors
    nb.config.NUMBA_COLOR_SCHEME = (
        "dark_bg"  # prints error messages for dark background terminals
    )
    nb.config.NUMBA_DEBUG_NRT = 1  # Numba run time (NRT) statistics counter
    nb.config.NUMBA_DEBUG_TYPEINFER = (
        1  # print out debugging information about type inference.
    )
    nb.config.NUMBA_ENABLE_PROFILING = 1  # enables profiler use
    nb.config.NUMBA_DUMP_CFG = 1  # prints out a control flow diagram
    nb.config.NUMBA_OPT = 0  # forums un optimized code from compilers
    nb.config.NUMBA_DEBUGINFO = 1  #
    nb.config.NUMBA_EXTEND_VARIABLE_LIFETIMES = (
        1  # allows for inspection of numba variables after end of compilation
    )

    # file="str.txt";file1="list.txt"
    # out=sys.stdout
    # sys.stdout=open('debug_numba_config.txt','w')
    # help(nb.config)
    # sys.stdout.close

    # print_msg('>> Numba config exported to debug_numba_config.txt')

# elif mode == "numba x86":
#    nb.config.NUMBA_OPT = 3
#    NUMBA_DISABLE_INTEL_SVML

import h5py
import numpy as np

from mpi4py import MPI
from scipy.stats import qmc

import mcdc.kernel as kernel
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.loop import (
    loop_fixed_source,
    loop_eigenvalue,
    loop_iqmc,
    set_cache,
    build_gpu_progs,
)
import mcdc.loop as loop
from mcdc.print_ import print_banner, print_msg, print_runtime, print_header_eigenvalue

# Get input_deck
import mcdc.global_ as mcdc_

input_deck = mcdc_.input_deck

import mcdc.adapt as adapt


def run():
    # Override input deck with command-line argument, if given
    if args.N_particle is not None:
        input_deck.setting["N_particle"] = args.N_particle
    if args.output is not None:
        input_deck.setting["output_name"] = args.output
    if args.progress_bar is not None:
        input_deck.setting["progress_bar"] = args.progress_bar

    # Start timer
    total_start = MPI.Wtime()

    # Preparation
    #   Set up and get the global variable container `mcdc` based on
    #   input deck
    preparation_start = MPI.Wtime()
    mcdc = prepare()
    mcdc["runtime_preparation"] = MPI.Wtime() - preparation_start

    # Print banner, hardware configuration, and header
    print_banner(mcdc)

    set_cache(mcdc["setting"]["caching"])

    print_msg(" Now running TNT...")
    if mcdc["setting"]["mode_eigenvalue"]:
        print_header_eigenvalue(mcdc)

    # Run simulation
    simulation_start = MPI.Wtime()
    if mcdc["technique"]["iQMC"]:
        loop_iqmc(mcdc)
    elif mcdc["setting"]["mode_eigenvalue"]:
        loop_eigenvalue(mcdc)
    else:
        loop_fixed_source(mcdc)
    mcdc["runtime_simulation"] = MPI.Wtime() - simulation_start

    # Output: generate hdf5 output files
    output_start = MPI.Wtime()
    generate_hdf5(mcdc)
    mcdc["runtime_output"] = MPI.Wtime() - output_start

    # Stop timer
    MPI.COMM_WORLD.Barrier()
    mcdc["runtime_total"] = MPI.Wtime() - total_start

    # Closout
    closeout(mcdc)


# =============================================================================
# utilities for handling discrepancies between input and state types
# =============================================================================


def copy_field(dst, src, name):
    if "padding" in name:
        return
    if isinstance(dst[name], np.ndarray):
        if isinstance(src[name], np.ndarray) and dst[name].shape != src[name].shape:
            for dim in src[name].shape:
                if dim == 0:
                    return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {src[name].shape}"
            )
        elif isinstance(src[name], list) and dst[name].shape[0] != len(src[name]):
            if len(src[name]) == 0:
                return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {len(src[name])}"
            )
    dst[name] = src[name]


# =============================================================================
# prepare domain decomposition
# =============================================================================
def get_d_idx(i, j, k, ni, nj):
    N = i + j * ni + k * ni * nj
    return N


def get_indexes(N, nx, ny):
    k = int(N / (nx * ny))
    j = int((N - nx * ny * k) / nx)
    i = int(N - nx * ny * k - nx * j)
    return i, j, k


def get_neighbors(N, w, nx, ny, nz):
    i, j, k = get_indexes(N, nx, ny)
    if i > 0:
        xn = get_d_idx(i - 1, j, k, nx, ny)
    else:
        xn = None
    if i < (nx - 1):
        xp = get_d_idx(i + 1, j, k, nx, ny)
    else:
        xp = None
    if j > 0:
        yn = get_d_idx(i, j - 1, k, nx, ny)
    else:
        yn = None
    if j < (ny - 1):
        yp = get_d_idx(i, j + 1, k, nx, ny)
    else:
        yp = None
    if k > 0:
        zn = get_d_idx(i, j, k - 1, nx, ny)
    else:
        zn = None
    if k < (nz - 1):
        zp = get_d_idx(i, j, k + 1, nx, ny)
    else:
        zp = None
    return xn, xp, yn, yp, zn, zp


def dd_prepare():
    work_ratio = input_deck.technique["dd_work_ratio"]

    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1

    input_deck.setting["bank_active_buff"] = 1000
    if input_deck.technique["dd_exchange_rate"] == None:
        input_deck.technique["dd_exchange_rate"] = 100

    if work_ratio is None:
        work_ratio = np.ones(d_Nx * d_Ny * d_Nz)
        input_deck.technique["dd_work_ratio"] = work_ratio

    if (
        input_deck.technique["domain_decomposition"]
        and np.sum(work_ratio) != MPI.COMM_WORLD.Get_size()
    ):
        print_msg(
            "Domain work ratio not equal to number of processors, %i != %i "
            % (np.sum(work_ratio), MPI.COMM_WORLD.Get_size())
        )
        exit()

    if input_deck.technique["domain_decomposition"]:
        # Assigning domain index
        i = 0
        rank_info = []
        for n in range(d_Nx * d_Ny * d_Nz):
            ranks = []
            for r in range(int(work_ratio[n])):
                ranks.append(i)
                if MPI.COMM_WORLD.Get_rank() == i:
                    d_idx = n
                i += 1
            rank_info.append(ranks)
        input_deck.technique["dd_idx"] = d_idx
        xn, xp, yn, yp, zn, zp = get_neighbors(d_idx, 0, d_Nx, d_Ny, d_Nz)
    else:
        input_deck.technique["dd_idx"] = 0
        input_deck.technique["dd_xp_neigh"] = []
        input_deck.technique["dd_xn_neigh"] = []
        input_deck.technique["dd_yp_neigh"] = []
        input_deck.technique["dd_yn_neigh"] = []
        input_deck.technique["dd_zp_neigh"] = []
        input_deck.technique["dd_zn_neigh"] = []
        return

    if xp is not None:
        input_deck.technique["dd_xp_neigh"] = rank_info[xp]
    else:
        input_deck.technique["dd_xp_neigh"] = []
    if xn is not None:
        input_deck.technique["dd_xn_neigh"] = rank_info[xn]
    else:
        input_deck.technique["dd_xn_neigh"] = []

    if yp is not None:
        input_deck.technique["dd_yp_neigh"] = rank_info[yp]
    else:
        input_deck.technique["dd_yp_neigh"] = []
    if yn is not None:
        input_deck.technique["dd_yn_neigh"] = rank_info[yn]
    else:
        input_deck.technique["dd_yn_neigh"] = []

    if zp is not None:
        input_deck.technique["dd_zp_neigh"] = rank_info[zp]
    else:
        input_deck.technique["dd_zp_neigh"] = []
    if zn is not None:
        input_deck.technique["dd_zn_neigh"] = rank_info[zn]
    else:
        input_deck.technique["dd_zn_neigh"] = []


def prepare():
    """
    Preparing the MC transport simulation:
      (1) Adapt kernels
      (2) Make types
      (3) Create and set up global variable container `mcdc`
    """

    dd_prepare()

    # =========================================================================
    # Adapt kernels
    # =========================================================================

    kernel.adapt_rng(nb.config.DISABLE_JIT)

    # =========================================================================
    # Make types
    # =========================================================================

    type_.make_type_particle(input_deck)
    type_.make_type_particle_record(input_deck)
    type_.make_type_nuclide(input_deck)
    type_.make_type_material(input_deck)
    type_.make_type_surface(input_deck)
    type_.make_type_cell(input_deck)
    type_.make_type_universe(input_deck)
    type_.make_type_lattice(input_deck)
    type_.make_type_source(input_deck)
    type_.make_type_tally(input_deck)
    type_.make_type_setting(input_deck)
    type_.make_type_uq_tally(input_deck)
    type_.make_type_uq(input_deck)
    type_.make_type_domain_decomp(input_deck)
    type_.make_type_dd_turnstile_event(input_deck)
    type_.make_type_technique(input_deck)
    type_.make_type_global(input_deck)
    kernel.adapt_rng(nb.config.DISABLE_JIT)

    type_.make_type_translate(input_deck)
    type_.make_type_group_array(input_deck)
    type_.make_type_j_array(input_deck)

    # =========================================================================
    # Create the global variable container
    #   TODO: Better alternative?
    # =========================================================================

    mcdc = np.zeros(1, dtype=type_.global_)[0]

    # Now, set up the global variable container

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # =========================================================================
    # Platform Targeting, Adapters, Toggles, etc
    # =========================================================================

    if target == "gpu":
        if not adapt.HAS_HARMONIZE:
            print_error(
                "No module named 'harmonize' - GPU functionality not available. "
            )
        adapt.gpu_forward_declare()

    adapt.set_toggle("iQMC", input_deck.technique["iQMC"])
    adapt.set_toggle("domain_decomp", input_deck.technique["domain_decomposition"])
    adapt.set_toggle("particle_tracker", mcdc["setting"]["track_particle"])
    adapt.eval_toggle()
    adapt.target_for(target)
    if target == "gpu":
        build_gpu_progs()
    adapt.nopython_mode(mode == "numba")

    # =========================================================================
    # Nuclides
    # =========================================================================

    N_nuclide = len(input_deck.nuclides)
    for i in range(N_nuclide):
        # General data
        for name in ["ID", "fissionable", "sensitivity", "sensitivity_ID", "dsm_Np"]:
            copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # MG data
        if mode_MG:
            for name in [
                "G",
                "J",
                "speed",
                "decay",
                "total",
                "capture",
                "scatter",
                "fission",
                "nu_s",
                "nu_f",
                "nu_p",
                "nu_d",
                "chi_s",
                "chi_p",
                "chi_d",
            ]:
                copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # CE data (load data from XS library)
        dir_name = os.getenv("MCDC_XSLIB")
        if mode_CE:
            nuc_name = input_deck.nuclides[i]["name"]
            with h5py.File(dir_name + "/" + nuc_name + ".h5", "r") as f:
                # Atomic weight ratio
                mcdc["nuclides"][i]["A"] = f["A"][()]
                # Energy grids
                for name in [
                    "E_xs",
                    "E_nu_p",
                    "E_nu_d",
                    "E_chi_p",
                    "E_chi_d1",
                    "E_chi_d2",
                    "E_chi_d3",
                    "E_chi_d4",
                    "E_chi_d5",
                    "E_chi_d6",
                ]:
                    mcdc["nuclides"][i]["N" + name] = len(f[name][:])
                    mcdc["nuclides"][i][name][: len(f[name][:])] = f[name][:]

                # XS
                for name in ["capture", "scatter", "fission"]:
                    mcdc["nuclides"][i]["ce_" + name][: len(f[name][:])] = f[name][:]
                    mcdc["nuclides"][i]["ce_total"][: len(f[name][:])] += f[name][:]

                # Fission production
                mcdc["nuclides"][i]["ce_nu_p"][: len(f["nu_p"][:])] = f["nu_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_nu_d"][j][: len(f["nu_d"][j, :])] = f[
                        "nu_d"
                    ][j, :]

                # Fission spectrum
                mcdc["nuclides"][i]["ce_chi_p"][: len(f["chi_p"][:])] = f["chi_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_chi_d%i" % (j + 1)][
                        : len(f["chi_d%i" % (j + 1)][:])
                    ] = f["chi_d%i" % (j + 1)][:]

                # Decay
                mcdc["nuclides"][i]["ce_decay"][: len(f["decay_rate"][:])] = f[
                    "decay_rate"
                ][:]

    # =========================================================================
    # Materials
    # =========================================================================

    N_material = len(input_deck.materials)
    for i in range(N_material):
        for name in type_.material.names:
            if name in ["nuclide_IDs", "nuclide_densities"]:
                mcdc["materials"][i][name][: mcdc["materials"][i]["N_nuclide"]] = (
                    input_deck.materials[i][name]
                )
            else:
                copy_field(mcdc["materials"][i], input_deck.materials[i], name)

    # =========================================================================
    # Surfaces
    # =========================================================================

    N_surface = len(input_deck.surfaces)
    for i in range(N_surface):
        for name in type_.surface.names:
            if name not in ["J", "t"]:
                copy_field(mcdc["surfaces"][i], input_deck.surfaces[i], name)

        # Variables with possible different sizes
        for name in ["J", "t"]:
            N = len(input_deck.surfaces[i][name])
            mcdc["surfaces"][i][name][:N] = input_deck.surfaces[i][name]

    # =========================================================================
    # Cells
    # =========================================================================

    N_cell = len(input_deck.cells)
    for i in range(N_cell):
        for name in type_.cell.names:
            if name not in ["surface_IDs", "positive_flags"]:
                copy_field(mcdc["cells"][i], input_deck.cells[i], name)

        # Variables with possible different sizes
        for name in ["surface_IDs", "positive_flags"]:
            N = len(input_deck.cells[i][name])
            mcdc["cells"][i][name][:N] = input_deck.cells[i][name]

    # =========================================================================
    # Universes
    # =========================================================================

    N_universe = len(input_deck.universes)
    for i in range(N_universe):
        for name in type_.universe.names:
            if name not in ["cell_IDs"]:
                mcdc["universes"][i][name] = input_deck.universes[i][name]

        # Variables with possible different sizes
        for name in ["cell_IDs"]:
            N = mcdc["universes"][i]["N_cell"]
            mcdc["universes"][i][name][:N] = input_deck.universes[i][name]

    # =========================================================================
    # Lattices
    # =========================================================================

    N_lattice = len(input_deck.lattices)
    for i in range(N_lattice):
        # Mesh
        for name in type_.mesh_uniform.names:
            mcdc["lattices"][i]["mesh"][name] = input_deck.lattices[i]["mesh"][name]

        # Universe IDs
        Nx = mcdc["lattices"][i]["mesh"]["Nx"]
        Ny = mcdc["lattices"][i]["mesh"]["Ny"]
        Nz = mcdc["lattices"][i]["mesh"]["Nz"]
        mcdc["lattices"][i]["universe_IDs"][:Nx, :Ny, :Nz] = input_deck.lattices[i][
            "universe_IDs"
        ]

    # =========================================================================
    # Source
    # =========================================================================

    N_source = len(input_deck.sources)
    for i in range(N_source):
        for name in type_.source.names:
            copy_field(mcdc["sources"][i], input_deck.sources[i], name)

    # Normalize source probabilities
    tot = 0.0
    for S in mcdc["sources"]:
        tot += S["prob"]
    for S in mcdc["sources"]:
        S["prob"] /= tot

    # =========================================================================
    # Tally
    # =========================================================================

    for name in type_.tally.names:
        if name not in ["score", "mesh"]:
            copy_field(mcdc["tally"], input_deck.tally, name)
    # Set mesh
    for name in type_.mesh_names:
        copy_field(mcdc["tally"]["mesh"], input_deck.tally["mesh"], name)

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        copy_field(mcdc["setting"], input_deck.setting, name)

    # Check if time boundary is above the final tally mesh time grid
    if mcdc["setting"]["time_boundary"] > mcdc["tally"]["mesh"]["t"][-1]:
        mcdc["setting"]["time_boundary"] = mcdc["tally"]["mesh"]["t"][-1]

    if input_deck.technique["iQMC"]:
        if len(mcdc["technique"]["iqmc"]["mesh"]["t"]) - 1 > 1:
            if (
                mcdc["setting"]["time_boundary"]
                > input_deck.technique["iqmc"]["mesh"]["t"][-1]
            ):
                mcdc["setting"]["time_boundary"] = input_deck.technique["iqmc"]["mesh"][
                    "t"
                ][-1]
    # =========================================================================
    # Technique
    # =========================================================================

    # Flags
    for name in [
        "weighted_emission",
        "implicit_capture",
        "population_control",
        "weight_window",
        "domain_decomposition",
        "weight_roulette",
        "iQMC",
        "IC_generator",
        "branchless_collision",
        "uq",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)

    # =========================================================================
    # Population control
    # =========================================================================

    # Population control technique (PCT)
    mcdc["technique"]["pct"] = input_deck.technique["pct"]
    mcdc["technique"]["pc_factor"] = input_deck.technique["pc_factor"]

    # =========================================================================
    # IC generator
    # =========================================================================

    for name in [
        "IC_N_neutron",
        "IC_N_precursor",
        "IC_neutron_density",
        "IC_neutron_density_max",
        "IC_precursor_density",
        "IC_precursor_density_max",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)

    # =========================================================================
    # Weight window (WW)
    # =========================================================================

    # WW mesh
    for name in type_.mesh_names[:-1]:
        copy_field(mcdc["technique"]["ww_mesh"], input_deck.technique["ww_mesh"], name)

    # WW windows
    mcdc["technique"]["ww"] = input_deck.technique["ww"]
    mcdc["technique"]["ww_width"] = input_deck.technique["ww_width"]

    # =========================================================================
    # Weight roulette
    # =========================================================================

    # Threshold
    mcdc["technique"]["wr_threshold"] = input_deck.technique["wr_threshold"]

    # Survival probability
    mcdc["technique"]["wr_survive"] = input_deck.technique["wr_survive"]
    # =========================================================================
    # Domain Decomposition
    # =========================================================================

    # Set domain mesh
    if input_deck.technique["domain_decomposition"]:
        for name in ["x", "y", "z", "t", "mu", "azi"]:
            copy_field(
                mcdc["technique"]["dd_mesh"], input_deck.technique["dd_mesh"], name
            )
        # Set exchange rate
        for name in ["dd_exchange_rate", "dd_repro"]:
            copy_field(mcdc["technique"], input_deck.technique, name)
        # Set domain index
        copy_field(mcdc, input_deck.technique, "dd_idx")
        for name in ["xp", "xn", "yp", "yn", "zp", "zn"]:
            copy_field(mcdc["technique"], input_deck.technique, f"dd_{name}_neigh")
        copy_field(mcdc["technique"], input_deck.technique, "dd_work_ratio")

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

    for name in type_.technique["iqmc"].names:
        if name not in [
            "mesh",
            "res",
            "lds",
            "generator",
            "sweep_counter",
            "total_source",
            "w_min",
            "score_list",
            "score",
        ]:
            copy_field(mcdc["technique"]["iqmc"], input_deck.technique["iqmc"], name)

    if input_deck.technique["iQMC"]:
        # pass in mesh
        iqmc = mcdc["technique"]["iqmc"]
        for name in ["x", "y", "z", "t"]:
            copy_field(iqmc["mesh"], input_deck.technique["iqmc"]["mesh"], name)
        # pass in score list
        for name, value in input_deck.technique["iqmc"]["score_list"].items():
            iqmc["score_list"][name] = value
        # pass in initial tallies
        for name, value in input_deck.technique["iqmc"]["score"].items():
            mcdc["technique"]["iqmc"]["score"][name] = value
        # LDS generator
        iqmc["generator"] = input_deck.technique["iqmc"]["generator"]
        # minimum particle weight
        iqmc["w_min"] = 1e-13
        # variables to generate samples
        scramble = iqmc["scramble"]
        N_dim = iqmc["N_dim"]
        seed = iqmc["seed"]
        N_particle = mcdc["setting"]["N_particle"]

        mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
        mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
        kernel.distribute_work(N_particle, mcdc)
        N_work = int(mcdc["mpi_work_size"])
        N_start = int(mcdc["mpi_work_start"])

        # generate LDS
        if input_deck.technique["iqmc"]["generator"] == "sobol":
            sampler = qmc.Sobol(d=N_dim, scramble=scramble)
            # skip the first entry in Sobol sequence because its 0.0
            # skip the second because it maps to ux = 0.0
            sampler.fast_forward(2)
            sampler.fast_forward(N_start)
            iqmc["lds"] = sampler.random(N_work)
        if input_deck.technique["iqmc"]["generator"] == "halton":
            sampler = qmc.Halton(d=N_dim, scramble=scramble, seed=seed)
            # skip the first entry in Halton sequence because its 0.0
            sampler.fast_forward(1)
            sampler.fast_forward(N_start)
            iqmc["lds"] = sampler.random(N_work)
        if input_deck.technique["iqmc"]["generator"] == "random":
            np.random.seed(seed)
            seeds = np.random.randint(1e6, size=mcdc["mpi_size"])
            np.random.seed(seeds[mcdc["mpi_rank"]])
            iqmc["lds"] = np.random.random((N_work, N_dim))

    # =========================================================================
    # Derivative Source Method
    # =========================================================================

    # Threshold
    mcdc["technique"]["dsm_order"] = input_deck.technique["dsm_order"]

    # =========================================================================
    # Variance Deconvolution - UQ
    # =========================================================================
    if mcdc["technique"]["uq"]:
        # Assumes that all tallies will also be uq tallies
        for name in type_.uq_tally.names:
            if name != "score":
                copy_field(mcdc["technique"]["uq_tally"], input_deck.tally, name)

        M = len(input_deck.uq_deltas["materials"])
        for i in range(M):
            idm = input_deck.uq_deltas["materials"][i]["ID"]
            mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"] = idm
            mcdc["technique"]["uq_"]["materials"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["materials"][i]["distribution"]
            )
            for name in input_deck.uq_deltas["materials"][i]["flags"]:
                mcdc["technique"]["uq_"]["materials"][i]["flags"][name] = True
                mcdc["technique"]["uq_"]["materials"][i]["delta"][name] = (
                    input_deck.uq_deltas["materials"][i][name]
                )
            flags = mcdc["technique"]["uq_"]["materials"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
                flags["speed"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True
            if mcdc["materials"][idm]["N_nuclide"] > 1:
                for name in type_.uq_mat.names:
                    mcdc["technique"]["uq_"]["materials"][i]["mean"][name] = (
                        input_deck.materials[idm][name]
                    )

        N = len(input_deck.uq_deltas["nuclides"])
        for i in range(N):
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["nuclides"][i]["distribution"]
            )
            idn = input_deck.uq_deltas["nuclides"][i]["ID"]
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"] = idn
            for name in type_.uq_nuc.names:
                copy_field(
                    mcdc["technique"]["uq_"]["nuclides"][i]["mean"],
                    input_deck.nuclides[idn],
                    name,
                )
            for name in input_deck.uq_deltas["nuclides"][i]["flags"]:
                if "padding" in name:
                    continue
                mcdc["technique"]["uq_"]["nuclides"][i]["flags"][name] = True
                copy_field(
                    mcdc["technique"]["uq_"]["nuclides"][i]["delta"],
                    input_deck.uq_deltas["nuclides"][i],
                    name,
                )
            flags = mcdc["technique"]["uq_"]["nuclides"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True

    # =========================================================================
    # MPI
    # =========================================================================

    # MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Distribute work to MPI ranks
    if mcdc["technique"]["domain_decomposition"]:
        kernel.distribute_work_dd(mcdc["setting"]["N_particle"], mcdc)
    else:
        kernel.distribute_work(mcdc["setting"]["N_particle"], mcdc)

    # =========================================================================
    # Particle banks
    # =========================================================================

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"

    # IC generator banks
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"

    # =========================================================================
    # Eigenvalue (or fixed-source)
    # =========================================================================

    # Initial guess
    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # Activate tally scoring for fixed-source
    if not mcdc["setting"]["mode_eigenvalue"]:
        mcdc["cycle_active"] = True

    # All active eigenvalue cycle?
    elif mcdc["setting"]["N_inactive"] == 0:
        mcdc["cycle_active"] = True

    # =========================================================================
    # Source file
    # =========================================================================

    if mcdc["setting"]["source_file"]:
        with h5py.File(mcdc["setting"]["source_file_name"], "r") as f:
            # Get source particle size
            N_particle = f["particles_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["particles"][start:end]
            mcdc["bank_source"]["size"] = N_local

    # =========================================================================
    # IC file
    # =========================================================================

    if mcdc["setting"]["IC_file"]:
        with h5py.File(mcdc["setting"]["IC_file_name"], "r") as f:
            # =================================================================
            # Set neutron source
            # =================================================================

            # Get source particle size
            N_particle = f["IC/neutrons_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["IC/neutrons"][start:end]
            mcdc["bank_source"]["size"] = N_local

            # =================================================================
            # Set precursor source
            # =================================================================

            # Get source particle size
            N_precursor = f["IC/precursors_size"][()]

            # Redistribute work
            kernel.distribute_work(N_precursor, mcdc, True)  # precursor = True
            N_local = mcdc["mpi_work_size_precursor"]
            start = mcdc["mpi_work_start_precursor"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_precursor"]["precursors"][:N_local] = f["IC/precursors"][
                start:end
            ]
            mcdc["bank_precursor"]["size"] = N_local

            # Set precursor strength
            if N_precursor > 0 and N_particle > 0:
                mcdc["precursor_strength"] = mcdc["bank_precursor"]["precursors"][0][
                    "w"
                ]

    loop.setup_gpu(mcdc)

    return mcdc


def dictlist_to_h5group(dictlist, input_group, name):
    main_group = input_group.create_group(name + "s")
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        elif v is None:
            next
        else:
            group[k] = v


def generate_hdf5(mcdc):
    if mcdc["mpi_master"]:
        if mcdc["setting"]["progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "w") as f:
            # Input deck
            if mcdc["setting"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                dictlist_to_h5group(input_deck.nuclides, input_group, "nuclide")
                dictlist_to_h5group(input_deck.materials, input_group, "material")
                dictlist_to_h5group(input_deck.surfaces, input_group, "surface")
                dictlist_to_h5group(input_deck.cells, input_group, "cell")
                dictlist_to_h5group(input_deck.universes, input_group, "universe")
                dictlist_to_h5group(input_deck.lattices, input_group, "lattice")
                dictlist_to_h5group(input_deck.sources, input_group, "source")
                dict_to_h5group(input_deck.tally, input_group.create_group("tally"))
                dict_to_h5group(input_deck.setting, input_group.create_group("setting"))
                dict_to_h5group(
                    input_deck.technique, input_group.create_group("technique")
                )

            # Tally
            T = mcdc["tally"]
            f.create_dataset("tally/grid/t", data=T["mesh"]["t"])
            f.create_dataset("tally/grid/x", data=T["mesh"]["x"])
            f.create_dataset("tally/grid/y", data=T["mesh"]["y"])
            f.create_dataset("tally/grid/z", data=T["mesh"]["z"])
            f.create_dataset("tally/grid/mu", data=T["mesh"]["mu"])
            f.create_dataset("tally/grid/azi", data=T["mesh"]["azi"])
            f.create_dataset("tally/grid/g", data=T["mesh"]["g"])

            # Scores
            for name in T["score"].dtype.names:
                if mcdc["tally"][name]:
                    name_h5 = name.replace("_", "-")
                    f.create_dataset(
                        "tally/" + name_h5 + "/mean",
                        data=np.squeeze(T["score"][name]["mean"]),
                    )
                    f.create_dataset(
                        "tally/" + name_h5 + "/sdev",
                        data=np.squeeze(T["score"][name]["sdev"]),
                    )
                    if mcdc["technique"]["uq_tally"][name]:
                        mc_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_var"
                        ]
                        tot_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_bin"
                        ]
                        f.create_dataset(
                            "tally/" + name_h5 + "/uq_var",
                            data=np.squeeze(tot_var - mc_var),
                        )

            # Eigenvalues
            if mcdc["setting"]["mode_eigenvalue"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                else:
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["setting"]["gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # dump iQMC mesh
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc"]["mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc"]["mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc"]["mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc"]["mesh"]["z"])
                # dump x,y,z scalar flux across all groups
                f.create_dataset(
                    "iqmc/tally/flux", data=np.squeeze(T["iqmc"]["score"]["flux"])
                )
                f.create_dataset(
                    "iqmc/tally/fission_source",
                    data=T["iqmc"]["score"]["fission-source"],
                )
                f.create_dataset(
                    "iqmc/tally/fission_power", data=T["iqmc"]["score"]["fission-power"]
                )
                f.create_dataset("iqmc/tally/source_constant", data=T["iqmc"]["source"])
                f.create_dataset(
                    "iqmc/tally/source_x", data=T["iqmc"]["score"]["tilt-x"]
                )
                f.create_dataset(
                    "iqmc/tally/source_y", data=T["iqmc"]["score"]["tilt-y"]
                )
                f.create_dataset(
                    "iqmc/tally/source_z", data=T["iqmc"]["score"]["tilt-z"]
                )
                # iteration data
                f.create_dataset("iqmc/itteration_count", data=T["iqmc"]["itt"])
                f.create_dataset("iqmc/final_residual", data=T["iqmc"]["res"])
                f.create_dataset("iqmc/sweep_count", data=T["iqmc"]["sweep_counter"])
                if mcdc["setting"]["mode_eigenvalue"]:
                    f.create_dataset(
                        "iqmc/outter_itteration_count", data=T["iqmc"]["itt_outter"]
                    )
                    f.create_dataset(
                        "iqmc/outter_final_residual", data=T["iqmc"]["res_outter"]
                    )

            # Particle tracker
            if mcdc["setting"]["track_particle"]:
                with h5py.File(mcdc["setting"]["output"] + "_ptrack.h5", "w") as f:
                    N_track = mcdc["particle_track_N"][0]
                    f.create_dataset("tracks", data=mcdc["particle_track"][:N_track])

            # IC generator
            if mcdc["technique"]["IC_generator"]:
                Nn = mcdc["technique"]["IC_bank_neutron"]["size"]
                Np = mcdc["technique"]["IC_bank_precursor"]["size"]
                f.create_dataset(
                    "IC/neutrons",
                    data=mcdc["technique"]["IC_bank_neutron"]["particles"][:Nn],
                )
                f.create_dataset(
                    "IC/precursors",
                    data=mcdc["technique"]["IC_bank_precursor"]["precursors"][:Np],
                )
                f.create_dataset("IC/neutrons_size", data=Nn)
                f.create_dataset("IC/precursors_size", data=Np)
                f.create_dataset(
                    "IC/fission", data=mcdc["technique"]["IC_fission"] / Nn
                )

    # Save particle?
    if mcdc["setting"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def closeout(mcdc):

    loop.teardown_gpu(mcdc)

    # Runtime
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

    print_runtime(mcdc)
    input_deck.reset()
