import h5py
import math
import numpy as np
import os

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

from mcdc.print_ import print_error


# ==============================================================================
# Basic types
# ==============================================================================

float64 = np.float64
int64 = np.int64
int32 = np.int32
uint64 = np.uint64
uint8 = np.uint8
bool_ = np.bool_
uintp = np.uintp
str_ = "U32"


# ==============================================================================
# MC/DC types
# ==============================================================================
# Currently defined based on input deck
# TODO: This causes JIT recompilation in certain cases

particle = None
particle_record = None
nuclide = None
material = None
surface = None
cell = None
universe = None
lattice = None
source = None
setting = None
tally = None
technique = None

# GPU mode related
translate = None
group_array = None
j_array = None
global_ = None


# ==============================================================================
# Alignment Logic
# ==============================================================================
# While CPU execution can robustly handle all sorts of Numba types, GPU
# execution requires structs to follow some of the basic properties expected of
# C-style structs with standard layout:
#
#      - Every primitive field is aligned by its size, and padding is inserted
#        between fields to ensure alignment in arrays and nested data structures
#
#      - Every field has a unique address
#
# If these rules are violated, memory accesses made in GPUs may encounter
# problems. For example, in cases where an access is not at an address aligned
# by their size, a segfault or similar fault will occur, or information will be
# lost. These issues were fixed by providing a function, align, which ensures the
# field lists fed to np.dtype fulfill these requirements.
#
# The align function does the following:
#
#      - Tracks the cumulative offset of fields as they appear in the input list.
#
#      - Inserts additional padding fields to ensure that primitive fields are
#        aligned by their size
#
#      - Re-sizes arrays to have at least one element in their array (this ensure
#        they have a non-zero size, and hence cannot overlap base addresses with
#        other fields.
#


def fixup_dims(dim_tuple):
    return tuple([max(d, 1) for d in dim_tuple])


def align(field_list):
    result = []
    offset = 0
    pad_id = 0
    for field in field_list:
        if len(field) > 3:
            print_error(
                "Unexpected struct field specification. Specifications \
                        usually only consist of 3 or fewer members"
            )
        multiplier = 1
        if len(field) == 3:
            field = (field[0], field[1], fixup_dims(field[2]))
            for d in field[2]:
                multiplier *= d
        kind = np.dtype(field[1])
        size = kind.itemsize

        if kind.isbuiltin == 0:
            alignment = 8
        elif kind.isbuiltin == 1:
            alignment = size
        else:
            print_error("Unexpected field item type")

        size *= multiplier

        if offset % alignment != 0:
            pad_size = alignment - (offset % alignment)
            result.append((f"padding_{pad_id}", uint8, (pad_size,)))
            pad_id += 1
            offset += pad_size

        result.append(field)
        offset += size

    if offset % 8 != 0:
        pad_size = 8 - (offset % 8)
        result.append((f"padding_{pad_id}", uint8, (pad_size,)))
        pad_id += 1

    return result


def into_dtype(field_list):
    result = np.dtype(align(field_list), align=True)
    return result


# ==============================================================================
# Copy Logic
# ==============================================================================


type_roster = {}


def copy_fn_for(kind, name):
    code = f"@njit\ndef copy_{name}(dst,src):\n"
    for f_name, spec in kind.fields.items():
        f_dtype = spec[0]
        if f_dtype in type_roster:
            kind_name = type_roster[f_dtype]["name"]
            code += f"    copy_{kind_name}(dst['{f_name}'],src['{f_name}'])"
        else:
            code += f"    dst['{f_name}'] = src['{f_name}']\n"
    type_roster[kind] = {}
    type_roster[kind]["name"] = name
    exec(code)
    return eval(f"copy_{name}")


# ==============================================================================
# Particle
# ==============================================================================


# Particle (in-flight)
def make_type_particle(input_deck):
    global particle

    struct = [
        ("x", float64),
        ("y", float64),
        ("z", float64),
        ("t", float64),
        ("ux", float64),
        ("uy", float64),
        ("uz", float64),
        ("g", uint64),
        ("E", float64),
        ("w", float64),
        ("alive", bool_),
        ("fresh", bool_),
        ("material_ID", int64),
        ("cell_ID", int64),
        ("surface_ID", int64),
        ("translation", float64, (3,)),
        ("event", int64),
        ("sensitivity_ID", int64),
        ("rng_seed", uint64),
    ]

    # Get modes
    iQMC = input_deck.technique["iQMC"]

    # =========================================================================
    # iQMC
    # =========================================================================

    # Default number of groups for iQMC
    G = 1

    # iQMC vector of weights
    if iQMC:
        G = input_deck.materials[0]["G"]
    iqmc_struct = [("w", float64, (G,))]
    struct += [("iqmc", iqmc_struct)]

    # Save type
    particle = into_dtype(struct)


# Particle record (in-bank)
def make_type_particle_record(input_deck):
    global particle_record, particle_record_mpi

    struct = [
        ("x", float64),
        ("y", float64),
        ("z", float64),
        ("t", float64),
        ("ux", float64),
        ("uy", float64),
        ("uz", float64),
        ("g", uint64),
        ("E", float64),
        ("w", float64),
        ("sensitivity_ID", int64),
        ("rng_seed", uint64),
    ]

    # Get modes
    iQMC = input_deck.technique["iQMC"]

    # =========================================================================
    # iQMC
    # =========================================================================

    # Default number of groups for iQMC
    G = 1

    # iQMC vector of weights
    if iQMC:
        G = input_deck.materials[0]["G"]
    iqmc_struct = [("w", float64, (G,))]
    struct += [("iqmc", iqmc_struct)]

    # Save type
    particle_record = into_dtype(struct)

    particle_record_mpi = from_numpy_dtype(particle_record)
    particle_record_mpi.Commit()


precursor = into_dtype(
    [
        ("x", float64),
        ("y", float64),
        ("z", float64),
        ("g", uint64),
        ("n_g", uint64),
        ("w", float64),
    ]
)


# ==============================================================================
# Particle bank
# ==============================================================================


def particle_bank(max_size):
    return into_dtype(
        [
            ("particles", particle_record, (max_size,)),
            ("size", int64, (1,)),
            ("tag", str_),
        ]
    )


def precursor_bank(max_size):
    return into_dtype(
        [("precursors", precursor, (max_size,)), ("size", int64, (1,)), ("tag", str_)]
    )


# ==============================================================================
# Nuclide
# ==============================================================================


def make_type_nuclide(input_deck):
    global nuclide

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # Get CE sizes
    if mode_CE:
        # Zeros for MG sizes
        G = 1
        J = 0

        # Get maximum energy grid sizes for CE data
        NE_xs = 0
        NE_nu_p = 0
        NE_nu_d = 0
        NE_chi_p = 0
        NE_chi_d1 = 0
        NE_chi_d2 = 0
        NE_chi_d3 = 0
        NE_chi_d4 = 0
        NE_chi_d5 = 0
        NE_chi_d6 = 0

        dir_name = os.getenv("MCDC_XSLIB")
        for nuc in input_deck.nuclides:
            with h5py.File(dir_name + "/" + nuc["name"] + ".h5", "r") as f:
                NE_xs = max(NE_xs, len(f["E_xs"][:]))
                NE_nu_p = max(NE_nu_p, len(f["E_nu_p"][:]))
                NE_nu_d = max(NE_nu_d, len(f["E_nu_d"][:]))
                NE_chi_p = max(NE_chi_p, len(f["E_chi_p"][:]))
                NE_chi_d1 = max(NE_chi_d1, len(f["E_chi_d1"][:]))
                NE_chi_d2 = max(NE_chi_d2, len(f["E_chi_d2"][:]))
                NE_chi_d3 = max(NE_chi_d3, len(f["E_chi_d3"][:]))
                NE_chi_d4 = max(NE_chi_d4, len(f["E_chi_d4"][:]))
                NE_chi_d5 = max(NE_chi_d5, len(f["E_chi_d5"][:]))
                NE_chi_d6 = max(NE_chi_d6, len(f["E_chi_d6"][:]))

    # Get MG sizes
    if mode_MG:
        G = input_deck.materials[0]["G"]
        J = input_deck.materials[0]["J"]

        # Zeros for CE sizes
        NE_xs = 0
        NE_nu_p = 0
        NE_nu_d = 0
        NE_chi_p = 0
        NE_chi_d1 = 0
        NE_chi_d2 = 0
        NE_chi_d3 = 0
        NE_chi_d4 = 0
        NE_chi_d5 = 0
        NE_chi_d6 = 0

    # General data
    struct = [
        ("ID", int64),
        ("fissionable", bool_),
        ("sensitivity", bool_),
        ("sensitivity_ID", int64),
        ("dsm_Np", float64),
        ("uq", bool_),
    ]

    # MG data
    struct += [
        ("G", int64),
        ("J", int64),
        ("speed", float64, (G,)),
        ("decay", float64, (J,)),
        ("total", float64, (G,)),
        ("capture", float64, (G,)),
        ("scatter", float64, (G,)),
        ("fission", float64, (G,)),
        ("nu_s", float64, (G,)),
        ("nu_f", float64, (G,)),
        ("nu_p", float64, (G,)),
        ("nu_d", float64, (G, J)),
        ("chi_s", float64, (G, G)),
        ("chi_p", float64, (G, G)),
        ("chi_d", float64, (J, G)),
    ]

    # CE data
    struct += [
        ("A", float64),
        ("NE_xs", int64),
        ("NE_nu_p", int64),
        ("NE_nu_d", int64),
        ("NE_chi_p", int64),
        ("NE_chi_d1", int64),
        ("NE_chi_d2", int64),
        ("NE_chi_d3", int64),
        ("NE_chi_d4", int64),
        ("NE_chi_d5", int64),
        ("NE_chi_d6", int64),
        ("E_xs", float64, (NE_xs,)),
        ("E_nu_p", float64, (NE_nu_p,)),
        ("E_nu_d", float64, (NE_nu_d,)),
        ("E_chi_p", float64, (NE_chi_p,)),
        ("E_chi_d1", float64, (NE_chi_d1,)),
        ("E_chi_d2", float64, (NE_chi_d2,)),
        ("E_chi_d3", float64, (NE_chi_d3,)),
        ("E_chi_d4", float64, (NE_chi_d4,)),
        ("E_chi_d5", float64, (NE_chi_d5,)),
        ("E_chi_d6", float64, (NE_chi_d6,)),
        ("ce_total", float64, (NE_xs,)),
        ("ce_capture", float64, (NE_xs,)),
        ("ce_scatter", float64, (NE_xs,)),
        ("ce_fission", float64, (NE_xs,)),
        ("ce_nu_p", float64, (NE_nu_p,)),
        ("ce_nu_d", float64, (6, NE_nu_d)),
        ("ce_chi_p", float64, (NE_chi_p,)),
        ("ce_chi_d1", float64, (NE_chi_d1,)),
        ("ce_chi_d2", float64, (NE_chi_d2,)),
        ("ce_chi_d3", float64, (NE_chi_d3,)),
        ("ce_chi_d4", float64, (NE_chi_d4,)),
        ("ce_chi_d5", float64, (NE_chi_d5,)),
        ("ce_chi_d6", float64, (NE_chi_d6,)),
        ("ce_decay", float64, (6,)),
    ]

    # Set the type
    nuclide = into_dtype(struct)


# ==============================================================================
# Material
# ==============================================================================


def make_type_material(input_deck):
    global material

    # Maximum number of nuclides per material
    Nmax_nuclide = max([material["N_nuclide"] for material in input_deck.materials])

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # Get CE sizes
    if mode_CE:
        # Zeros for MG sizes
        G = 1
        J = 0

    # Get MG sizes
    if mode_MG:
        G = input_deck.materials[0]["G"]
        J = input_deck.materials[0]["J"]

    # General data
    struct = [
        ("ID", int64),
        ("N_nuclide", int64),
        ("sensitivity", bool_),
        ("nuclide_IDs", int64, (Nmax_nuclide,)),
        ("nuclide_densities", float64, (Nmax_nuclide,)),
        ("uq", bool_),
    ]

    # MG data
    struct += [
        ("G", int64),
        ("J", int64),
        ("speed", float64, (G,)),
        ("total", float64, (G,)),
        ("capture", float64, (G,)),
        ("scatter", float64, (G,)),
        ("fission", float64, (G,)),
        ("nu_s", float64, (G,)),
        ("nu_f", float64, (G,)),
        ("nu_p", float64, (G,)),
        ("nu_d", float64, (G, J)),
        ("chi_s", float64, (G, G)),
        ("chi_p", float64, (G, G)),
    ]

    # Set the type
    material = into_dtype(struct)


# ==============================================================================
# Surface
# ==============================================================================


def make_type_surface(input_deck):
    global surface

    # Maximum number of time-dependent surface slices
    Nmax_slice = 0
    for surface in input_deck.surfaces:
        Nmax_slice = max(Nmax_slice, surface["N_slice"])

    surface = into_dtype(
        [
            ("ID", int64),
            ("N_slice", int64),
            ("vacuum", bool_),
            ("reflective", bool_),
            ("A", float64),
            ("B", float64),
            ("C", float64),
            ("D", float64),
            ("E", float64),
            ("F", float64),
            ("G", float64),
            ("H", float64),
            ("I", float64),
            ("J", float64, (Nmax_slice, 2)),
            ("t", float64, (Nmax_slice + 1,)),
            ("linear", bool_),
            ("nx", float64),
            ("ny", float64),
            ("nz", float64),
            ("sensitivity", bool_),
            ("sensitivity_ID", int64),
            ("dsm_Np", float64),
        ]
    )


# ==============================================================================
# Cell
# ==============================================================================


def make_type_cell(input_deck):
    global cell

    # Maximum number of surfaces per cell
    Nmax_surface = max([cell["N_surface"] for cell in input_deck.cells])

    cell = into_dtype(
        [
            ("ID", int64),
            ("N_surface", int64),
            ("surface_IDs", int64, (Nmax_surface,)),
            ("positive_flags", bool_, (Nmax_surface,)),
            ("material_ID", int64),
            ("lattice", bool_),
            ("lattice_ID", int64),
            ("lattice_center", float64, (3,)),
        ]
    )


# ==============================================================================
# Universe
# ==============================================================================


def make_type_universe(input_deck):
    global universe

    # Maximum number of cells per universe
    Nmax_cell = max([universe["N_cell"] for universe in input_deck.universes])

    # Default root universe, if not defined
    N_cell = len(input_deck.cells)
    N_universe = len(input_deck.universes)
    if N_universe == 1:
        Nmax_cell = N_cell
        card = input_deck.universes[0]
        card["N_cell"] = N_cell
        card["cell_IDs"] = np.arange(N_cell)

    universe = into_dtype(
        [("ID", int64), ("N_cell", int64), ("cell_IDs", int64, (Nmax_cell,))]
    )


# ==============================================================================
# Lattice
# ==============================================================================


mesh_uniform = into_dtype(
    [
        ("x0", float64),
        ("dx", float64),
        ("Nx", int64),
        ("y0", float64),
        ("dy", float64),
        ("Ny", int64),
        ("z0", float64),
        ("dz", float64),
        ("Nz", int64),
    ]
)


def make_type_lattice(input_deck):
    global lattice

    # Max dimensional grids
    Nmax_x = 0
    Nmax_y = 0
    Nmax_z = 0
    for card in input_deck.lattices:
        Nmax_x = max(Nmax_x, card["mesh"]["Nx"])
        Nmax_y = max(Nmax_y, card["mesh"]["Ny"])
        Nmax_z = max(Nmax_z, card["mesh"]["Nz"])

    lattice = into_dtype(
        [("mesh", mesh_uniform), ("universe_IDs", int64, (Nmax_x, Nmax_y, Nmax_z))]
    )


# ==============================================================================
# Source
# ==============================================================================


def make_type_source(input_deck):
    global source

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # Get energy data size
    if mode_CE:
        G = 1
        # Maximum number of data point in energy pdf
        Nmax_E = max([source["energy"].shape[1] for source in input_deck.sources])
    if mode_MG:
        G = input_deck.materials[0]["G"]
        Nmax_E = 2

    # General data
    struct = [
        ("ID", int64),
        ("box", bool_),
        ("isotropic", bool_),
        ("white", bool_),
        ("x", float64),
        ("y", float64),
        ("z", float64),
        ("box_x", float64, (2,)),
        ("box_y", float64, (2,)),
        ("box_z", float64, (2,)),
        ("ux", float64),
        ("uy", float64),
        ("uz", float64),
        ("white_x", float64),
        ("white_y", float64),
        ("white_z", float64),
        ("time", float64, (2,)),
        ("prob", float64),
    ]

    # MG data
    struct += [
        ("group", float64, (G,)),
    ]

    # CE data
    struct += [
        ("energy", float64, (2, Nmax_E)),
    ]

    source = into_dtype(struct)


# ==============================================================================
# Tally
# ==============================================================================


# Score lists
score_list = (
    "flux",
    "density",
    "fission",
    "total",
    "current",
    "eddington",
    "exit",
)


def make_type_tally(input_deck):
    global tally

    # Number of sensitivitys parameters
    N_sensitivity = input_deck.setting["N_sensitivity"]

    # Number of tally scores
    Ns = 1 + N_sensitivity
    if input_deck.technique["dsm_order"] == 2:
        Ns = 1 + 2 * N_sensitivity + int(0.5 * N_sensitivity * (N_sensitivity - 1))

    # Get card
    card = input_deck.tally

    # Tally estimator flags
    struct = [("tracklength", bool_)]

    def make_type_score(shape):
        return into_dtype(
            [
                ("bin", float64, shape),
                ("mean", float64, shape),
                ("sdev", float64, shape),
            ]
        )

    # Mesh
    mesh, Nx, Ny, Nz, Nt, Nmu, N_azi, Ng = make_type_mesh(card["mesh"])
    struct += [("mesh", mesh)]

    # Scores and shapes
    scores_shapes = [
        ["flux", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["density", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["fission", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["total", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["current", (Ns, Ng, Nt, Nx, Ny, Nz, 3)],
        ["eddington", (Ns, Ng, Nt, Nx, Ny, Nz, 6)],
        ["exit", (Ns, Ng, Nt, 2, Ny, Nz, Nmu, N_azi)],
    ]

    # Add score flags to structure
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        struct += [(name, bool_)]

    # Add scores to structure
    scores_struct = []
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        shape = scores_shapes[i][1]
        if not card[name]:
            shape = (0,) * len(shape)
        scores_struct += [(name, make_type_score(shape))]
    scores = into_dtype(scores_struct)
    struct += [("score", scores)]

    # Make tally structure
    tally = into_dtype(struct)


# ==============================================================================
# Setting
# ==============================================================================


def make_type_setting(deck):
    global setting

    card = deck.setting
    struct = [
        # Basic MC simulation parameters
        ("N_particle", uint64),
        ("N_batch", uint64),
        ("rng_seed", uint64),
        ("time_boundary", float64),
        # Physics flags
        ("mode_MG", bool_),
        ("mode_CE", bool_),
        # Misc.
        ("progress_bar", bool_),
        ("caching", bool_),
        ("output_name", str_),
        ("save_input_deck", bool_),
        ("track_particle", bool_),
        # Eigenvalue mode
        ("mode_eigenvalue", bool_),
        ("k_init", float64),
        ("N_inactive", uint64),
        ("N_active", uint64),
        ("N_cycle", uint64),
        ("save_particle", bool_),
        ("gyration_radius", bool_),
        ("gyration_radius_type", uint64),
        # Time census
        ("N_census", uint64),
        ("census_time", float64, (card["N_census"],)),
        # Particle source file
        ("source_file", bool_),
        ("source_file_name", str_),
        # Initial condition source file
        ("IC_file", bool_),
        ("IC_file_name", str_),
        ("N_precursor", uint64),
        # TODO: Move to technique
        ("N_sensitivity", uint64),
    ]

    # Finalize setting type
    setting = into_dtype(struct)


# ==============================================================================
# Technique
# ==============================================================================

iqmc_score_list = (
    "flux",
    "effective-scattering",
    "effective-fission",
    "tilt-x",
    "tilt-y",
    "tilt-z",
    "fission-power",
    "fission-source",
)


def make_type_technique(input_deck):
    global technique

    # Get sizes
    N_particle = input_deck.setting["N_particle"]

    # Get modes
    mode_MG = input_deck.setting["mode_MG"]

    # Get card
    card = input_deck.technique

    # Number of groups
    if mode_MG:
        G = input_deck.materials[0]["G"]
    else:
        G = 1

    # Technique flags
    struct = [
        ("weighted_emission", bool_),
        ("implicit_capture", bool_),
        ("population_control", bool_),
        ("weight_window", bool_),
        ("weight_roulette", bool_),
        ("iQMC", bool_),
        ("IC_generator", bool_),
        ("branchless_collision", bool_),
        ("domain_decomposition", bool_),
        ("uq", bool_),
    ]

    # =========================================================================
    # Population control
    # =========================================================================

    struct += [("pct", int64), ("pc_factor", float64)]

    # =========================================================================
    # domain decomp
    # =========================================================================
    # Mesh
    mesh, Nx, Ny, Nz, Nt, Nmu, N_azi, Ng = make_type_mesh(card["dd_mesh"])
    struct += [("dd_mesh", mesh)]
    struct += [("dd_idx", int64)]
    struct += [("dd_sent", int64)]
    struct += [("dd_work_ratio", int64, (len(card["dd_work_ratio"]),))]
    struct += [("dd_exchange_rate", int64)]
    struct += [("dd_repro", bool_)]
    struct += [("dd_xp_neigh", int64, (len(card["dd_xp_neigh"]),))]
    struct += [("dd_xn_neigh", int64, (len(card["dd_xn_neigh"]),))]
    struct += [("dd_yp_neigh", int64, (len(card["dd_yp_neigh"]),))]
    struct += [("dd_yn_neigh", int64, (len(card["dd_yn_neigh"]),))]
    struct += [("dd_zp_neigh", int64, (len(card["dd_zp_neigh"]),))]
    struct += [("dd_zn_neigh", int64, (len(card["dd_zn_neigh"]),))]

    # =========================================================================
    # Weight window
    # =========================================================================

    # Mesh
    mesh, Nx, Ny, Nz, Nt, Nmu, N_azi, Ng = make_type_mesh(card["ww_mesh"])
    struct += [("ww_mesh", mesh)]
    struct += [("ww_width", float64)]

    # Window
    struct += [("ww", float64, (Nt, Nx, Ny, Nz))]

    # =========================================================================
    # Weight Roulette
    # =========================================================================

    # Constants
    struct += [("wr_threshold", float64), ("wr_survive", float64)]

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================
    iqmc_list = []

    # Mesh (for qmc source tallies)
    if card["iQMC"]:
        mesh, Nx, Ny, Nz, Nt, Nmu, N_azi = make_type_mesh_(card["iqmc"]["mesh"])
        Ng = G
        N_dim = 6  # group, x, y, z, mu, phi
    else:
        Nx = Ny = Nz = Nt = Nmu = N_azi = N_particle = Ng = N_dim = 0

    iqmc_list += [("mesh", mesh)]

    # Low-discprenecy sequence
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # Evenly distribute work
    work_size = math.floor(N_particle / size)
    # Count reminder
    rem = N_particle % size
    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1

    iqmc_list += [("lds", float64, (work_size, N_dim))]
    iqmc_list += [("fixed_source", float64, (Ng, Nt, Nx, Ny, Nz))]
    # TODO: make matidx int32
    iqmc_list += [("material_idx", int64, (Nt, Nx, Ny, Nz))]
    # this is the original source matrix size + all tilted sources
    iqmc_list += [("source", float64, (Ng, Nt, Nx, Ny, Nz))]
    total_size = (Ng * Nt * Nx * Ny * Nz) * card["iqmc"]["krylov_vector_size"]
    iqmc_list += [(("total_source"), float64, (total_size,))]

    # Scores and shapes
    scores_shapes = [
        ["flux", (Ng, Nt, Nx, Ny, Nz)],
        ["effective-scattering", (Ng, Nt, Nx, Ny, Nz)],
        ["effective-fission", (Ng, Nt, Nx, Ny, Nz)],
        ["tilt-x", (Ng, Nt, Nx, Ny, Nz)],
        ["tilt-y", (Ng, Nt, Nx, Ny, Nz)],
        ["tilt-z", (Ng, Nt, Nx, Ny, Nz)],
        ["fission-power", (Ng, Nt, Nx, Ny, Nz)],  # SigmaF*phi
        ["fission-source", (1,)],  # nu*SigmaF*phi
    ]

    if card["iQMC"]:
        if setting["mode_eigenvalue"]:
            if card["iqmc"]["eigenmode_solver"] == "power_iteration":
                card["iqmc"]["score_list"]["fission-source"] = True

    # Add score flags to structure
    score_list = []
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        score_list += [(name, bool_)]
    score_list = into_dtype(score_list)
    iqmc_list += [("score_list", score_list)]

    # Add scores to structure
    scores_struct = []
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        shape = scores_shapes[i][1]
        if not card["iqmc"]["score_list"][name]:
            shape = (0,) * len(shape)
        scores_struct += [(name, float64, shape)]
    # TODO: make outter effective fission size zero if not eigenmode
    # (causes problems with numba)
    scores_struct += [("effective-fission-outter", float64, (Ng, Nt, Nx, Ny, Nz))]
    scores = into_dtype(scores_struct)
    iqmc_list += [("score", scores)]

    # Constants
    iqmc_list += [
        ("maxitt", int64),
        ("tol", float64),
        ("itt", int64),
        ("itt_outter", int64),
        ("res", float64),
        ("res_outter", float64),
        ("N_dim", int64),
        ("scramble", bool_),
        ("seed", int64),
        ("generator", str_),
        ("fixed_source_solver", str_),
        ("eigenmode_solver", str_),
        ("krylov_restart", int64),
        ("preconditioner_sweeps", int64),
        ("sweep_counter", int64),
        ("w_min", float64),
    ]

    struct += [("iqmc", into_dtype(iqmc_list))]

    # =========================================================================
    # IC generator
    # =========================================================================

    # Create bank types
    #   We need local banks to ensure reproducibility regardless of # of MPIs
    #   TODO: Having smaller bank buffer (~N_target/MPI_size) and even smaller
    #         local bank would be more efficient.
    if card["IC_generator"]:
        Nn = int(card["IC_N_neutron"] * 1.2)
        Np = int(card["IC_N_precursor"] * 1.2)
        Nn_local = Nn
        Np_local = Np
    else:
        Nn = 0
        Np = 0
        Nn_local = 0
        Np_local = 0
    bank_neutron = particle_bank(Nn)
    bank_neutron_local = particle_bank(Nn_local)
    bank_precursor = precursor_bank(Np)
    bank_precursor_local = precursor_bank(Np_local)

    # The parameters
    struct += [
        ("IC_N_neutron", int64),
        ("IC_N_precursor", int64),
        ("IC_neutron_density", float64),
        ("IC_neutron_density_max", float64),
        ("IC_precursor_density", float64),
        ("IC_precursor_density_max", float64),
        ("IC_bank_neutron_local", bank_neutron_local),
        ("IC_bank_precursor_local", bank_precursor_local),
        ("IC_bank_neutron", bank_neutron),
        ("IC_bank_precursor", bank_precursor),
        ("IC_fission_score", float64, (1,)),
        ("IC_fission", float64),
    ]

    # =========================================================================
    # Derivative Source Method
    # =========================================================================

    struct += [
        ("dsm_order", int64),
    ]

    # =========================================================================
    # Variance Deconvolution
    # =========================================================================
    struct += [("uq_tally", uq_tally), ("uq_", uq)]

    # Finalize technique type
    technique = into_dtype(struct)


# UQ
def make_type_uq_tally(input_deck):
    global uq_tally

    def make_type_uq_score(shape):
        return into_dtype(
            [
                ("batch_bin", float64, shape),
                ("batch_var", float64, shape),
            ]
        )

    # Tally estimator flags
    struct = []

    # Number of tally scores
    Ns = 1 + input_deck.setting["N_sensitivity"]

    # Tally card
    tally_card = input_deck.tally

    # Mesh, but doesn't need to be added
    mesh, Nx, Ny, Nz, Nt, Nmu, N_azi, Ng = make_type_mesh(tally_card["mesh"])

    # Scores and shapes
    scores_shapes = [
        ["flux", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["density", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["fission", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["total", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["current", (Ns, Ng, Nt, Nx, Ny, Nz, 3)],
        ["eddington", (Ns, Ng, Nt, Nx, Ny, Nz, 6)],
        ["exit", (Ns, Ng, Nt, 2, Ny, Nz, Nmu, N_azi)],
    ]

    # Add score flags to structure
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        struct += [(name, bool_)]

    # Add scores to structure
    scores_struct = []
    for i in range(len(scores_shapes)):
        name = scores_shapes[i][0]
        shape = scores_shapes[i][1]
        if not tally_card[name]:
            shape = (0,) * len(shape)
        scores_struct += [(name, make_type_uq_score(shape))]
    scores = into_dtype(scores_struct)
    struct += [("score", scores)]

    # Make tally structure
    uq_tally = into_dtype(struct)


def make_type_uq(input_deck):
    global uq, uq_nuc, uq_mat

    #    def make_type_parameter(shape):
    #        return into_dtype(
    #            [
    #                ("tag", str_),             # nuclides, materials, surfaces, sources
    #                ("ID", int64),
    #                ("key", str_),
    #                ("mean", float64, shape),
    #                ("delta", float64, shape),
    #                ("distribution", str_),
    #                ("rng_seed", uint64),
    #            ]
    #        )

    def make_type_parameter(G, J, decay=False):
        # Fields are things that can have deltas
        struct = [
            ("speed", float64, (G,)),
            ("capture", float64, (G,)),
            ("scatter", float64, (G, G)),
            ("fission", float64, (G,)),
            ("nu_s", float64, (G,)),
            ("nu_p", float64, (G,)),
            ("nu_d", float64, (G, J)),
            ("chi_p", float64, (G, G)),
        ]
        struct += [("decay", float64, (J,)), ("chi_d", float64, (J, G))]
        return into_dtype(struct)

    # Size numbers
    G = input_deck.materials[0]["G"]
    J = input_deck.materials[0]["J"]

    # UQ deck
    uq_deck = input_deck.uq_deltas

    uq_nuc = make_type_parameter(G, J, True)
    uq_mat = make_type_parameter(G, J)

    flags = into_dtype(
        [
            ("speed", bool_),
            ("decay", bool_),
            ("total", bool_),
            ("capture", bool_),
            ("scatter", bool_),
            ("fission", bool_),
            ("nu_s", bool_),
            ("nu_f", bool_),
            ("nu_p", bool_),
            ("nu_d", bool_),
            ("chi_s", bool_),
            ("chi_p", bool_),
            ("chi_d", bool_),
        ]
    )
    info = into_dtype([("distribution", str_), ("ID", int64), ("rng_seed", uint64)])

    container = into_dtype(
        [("mean", uq_nuc), ("delta", uq_mat), ("flags", flags), ("info", info)]
    )

    N_nuclide = len(uq_deck["nuclides"])
    N_material = len(uq_deck["materials"])
    uq = into_dtype(
        [("nuclides", container, (N_nuclide,)), ("materials", container, (N_material,))]
    )


def make_type_dd_turnstile_event(input_deck):
    global dd_turnstile_event, dd_turnstile_event_mpi
    dd_turnstile_event = into_dtype(
        [
            ("busy_delta", int32),
            ("send_delta", int32),
        ]
    )
    dd_turnstile_event_mpi = from_numpy_dtype(dd_turnstile_event)
    dd_turnstile_event_mpi.Commit()


def make_type_domain_decomp(input_deck):
    global domain_decomp
    # Domain banks if needed
    if input_deck.technique["domain_decomposition"]:
        bank_domain_xp = particle_bank(input_deck.technique["dd_exchange_rate"])
        bank_domain_xn = particle_bank(input_deck.technique["dd_exchange_rate"])
        bank_domain_yp = particle_bank(input_deck.technique["dd_exchange_rate"])
        bank_domain_yn = particle_bank(input_deck.technique["dd_exchange_rate"])
        bank_domain_zp = particle_bank(input_deck.technique["dd_exchange_rate"])
        bank_domain_zn = particle_bank(input_deck.technique["dd_exchange_rate"])
    else:
        bank_domain_xp = particle_bank(0)
        bank_domain_xn = particle_bank(0)
        bank_domain_yp = particle_bank(0)
        bank_domain_yn = particle_bank(0)
        bank_domain_zp = particle_bank(0)
        bank_domain_zn = particle_bank(0)

    domain_decomp = into_dtype(
        [
            # Info tracked in all ranks
            ("bank_xp", bank_domain_xp),
            ("bank_xn", bank_domain_xn),
            ("bank_yp", bank_domain_yp),
            ("bank_yn", bank_domain_yn),
            ("bank_zp", bank_domain_zp),
            ("bank_zn", bank_domain_zn),
            ("send_count", int64),  # Number of particles sent
            ("recv_count", int64),  # Number of particles recv'd
            ("rank_busy", bool_),  # True if the rank currently has particles to process
            (
                "work_done",
                int64,
            ),  # Whether or not there is any outstanding work across any ranks
            # Info tracked in "leader" rank zero
            (
                "send_total",
                int64,
            ),  # The total number of particles sent but not yet recv'd
            ("busy_total", int64),  # The total number of busy ranks
        ]
    )


param_names = ["tag", "ID", "key", "mean", "delta", "distribution", "rng_seed"]


# ==============================================================================
# Global
# ==============================================================================


def make_type_global(input_deck):
    global global_

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # Numbers of objects
    N_nuclide = len(input_deck.nuclides)
    N_material = len(input_deck.materials)
    N_surface = len(input_deck.surfaces)
    N_cell = len(input_deck.cells)
    N_source = len(input_deck.sources)
    N_universe = len(input_deck.universes)
    N_lattice = len(input_deck.lattices)

    # Simulation parameters
    N_particle = input_deck.setting["N_particle"]
    N_precursor = input_deck.setting["N_precursor"]
    N_cycle = input_deck.setting["N_cycle"]

    # Particle bank buffers
    bank_active_buff = input_deck.setting["bank_active_buff"]
    bank_census_buff = input_deck.setting["bank_census_buff"]

    # Number of precursor groups
    if mode_MG:
        J = input_deck.materials[0]["J"]
    if mode_CE:
        J = 6

    # Number of work
    N_work = math.ceil(N_particle / MPI.COMM_WORLD.Get_size())
    N_work_precursor = math.ceil(N_precursor / MPI.COMM_WORLD.Get_size())

    # Particle bank types
    bank_active = particle_bank(1 + bank_active_buff)
    if input_deck.setting["mode_eigenvalue"] or input_deck.setting["N_census"] > 1:
        bank_census = particle_bank(int((1 + bank_census_buff) * N_work))
        bank_source = particle_bank(int((1 + bank_census_buff) * N_work))
    else:
        bank_census = particle_bank(0)
        bank_source = particle_bank(0)
    bank_precursor = precursor_bank(0)

    # Particle tracker
    N_track = 0
    if input_deck.setting["track_particle"]:
        N_track = N_work * 1000

    # iQMC bank adjustment
    if input_deck.technique["iQMC"]:
        bank_source = particle_bank(N_work)
        if input_deck.setting["mode_eigenvalue"]:
            bank_census = particle_bank(0)

    # Source and IC files bank adjustments
    if not input_deck.setting["mode_eigenvalue"]:
        if input_deck.setting["source_file"]:
            bank_source = particle_bank(N_work)
        if input_deck.setting["IC_file"]:
            bank_source = particle_bank(N_work)
            bank_precursor = precursor_bank(N_precursor)

    if (
        input_deck.setting["source_file"] and not input_deck.setting["mode_eigenvalue"]
    ) or input_deck.technique["iQMC"]:
        bank_source = particle_bank(N_work)

    # GLobal type
    global_ = into_dtype(
        [
            ("nuclides", nuclide, (N_nuclide,)),
            ("materials", material, (N_material,)),
            ("surfaces", surface, (N_surface,)),
            ("cells", cell, (N_cell,)),
            ("universes", universe, (N_universe,)),
            ("lattices", lattice, (N_lattice,)),
            ("sources", source, (N_source,)),
            ("tally", tally),
            ("setting", setting),
            ("technique", technique),
            ("domain_decomp", domain_decomp),
            ("bank_active", bank_active),
            ("bank_census", bank_census),
            ("bank_source", bank_source),
            ("bank_precursor", bank_precursor),
            ("rng_seed_base", uint64),
            ("rng_seed", uint64),
            ("rng_stride", int64),
            ("dd_idx", int64),
            ("k_eff", float64),
            ("k_cycle", float64, (N_cycle,)),
            ("k_avg", float64),
            ("k_sdv", float64),
            ("n_avg", float64),  # Neutron density
            ("n_sdv", float64),
            ("n_max", float64),
            ("C_avg", float64),  # Precursor density
            ("C_sdv", float64),
            ("C_max", float64),
            ("k_avg_running", float64),
            ("k_sdv_running", float64),
            ("gyration_radius", float64, (N_cycle,)),
            ("idx_cycle", int64),
            ("cycle_active", bool_),
            ("eigenvalue_tally_nuSigmaF", float64, (1,)),
            ("eigenvalue_tally_n", float64, (1,)),
            ("eigenvalue_tally_C", float64, (1,)),
            ("idx_census", int64),
            ("idx_batch", int64),
            ("mpi_size", int64),
            ("mpi_rank", int64),
            ("mpi_master", bool_),
            ("mpi_work_start", int64),
            ("mpi_work_size", int64),
            ("mpi_work_size_total", int64),
            ("mpi_work_start_precursor", int64),
            ("mpi_work_size_precursor", int64),
            ("mpi_work_size_total_precursor", int64),
            ("runtime_total", float64),
            ("runtime_preparation", float64),
            ("runtime_simulation", float64),
            ("runtime_output", float64),
            ("runtime_bank_management", float64),
            ("particle_track", float64, (N_track, 8)),
            ("particle_track_N", int64, (1,)),
            ("particle_track_history_ID", int64, (1,)),
            ("particle_track_particle_ID", int64, (1,)),
            ("precursor_strength", float64),
            ("mpi_work_iter", int64, (1,)),
            ("gpu_state", uintp),
            ("source_program", uintp),
            ("precursor_program", uintp),
            ("source_seed", uint64),
        ]
    )


# ==============================================================================
# Util
# ==============================================================================


def make_type_translate(input_deck):
    global translate
    translate = into_dtype([("values", float64, (3,))])


def make_type_group_array(input_deck):
    global group_array
    G = input_deck.materials[0]["G"]
    group_array = into_dtype([("values", float64, (G,))])


def make_type_j_array(input_deck):
    global j_array
    J = input_deck.materials[0]["J"]
    j_array = into_dtype([("values", float64, (J,))])


def make_type_mesh(card):
    Nx = len(card["x"]) - 1
    Ny = len(card["y"]) - 1
    Nz = len(card["z"]) - 1
    Nt = len(card["t"]) - 1
    Nmu = len(card["mu"]) - 1
    N_azi = len(card["azi"]) - 1
    Ng = len(card["g"]) - 1
    return (
        into_dtype(
            [
                ("x", float64, (Nx + 1,)),
                ("y", float64, (Ny + 1,)),
                ("z", float64, (Nz + 1,)),
                ("t", float64, (Nt + 1,)),
                ("mu", float64, (Nmu + 1,)),
                ("azi", float64, (N_azi + 1,)),
                ("g", float64, (Ng + 1,)),
            ]
        ),
        Nx,
        Ny,
        Nz,
        Nt,
        Nmu,
        N_azi,
        Ng,
    )


def make_type_mesh_(card):
    Nx = len(card["x"]) - 1
    Ny = len(card["y"]) - 1
    Nz = len(card["z"]) - 1
    Nt = len(card["t"]) - 1
    Nmu = len(card["mu"]) - 1
    N_azi = len(card["azi"]) - 1
    return (
        into_dtype(
            [
                ("x", float64, (Nx + 1,)),
                ("y", float64, (Ny + 1,)),
                ("z", float64, (Nz + 1,)),
                ("t", float64, (Nt + 1,)),
                ("mu", float64, (Nmu + 1,)),
                ("azi", float64, (N_azi + 1,)),
            ]
        ),
        Nx,
        Ny,
        Nz,
        Nt,
        Nmu,
        N_azi,
    )


mesh_names = ["x", "y", "z", "t", "mu", "azi", "g"]
