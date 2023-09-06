import math
import numpy as np
import sys
from numba import njit
from mcdc.print_ import print_error

from mpi4py import MPI

# Basic types
float64 = np.float64
int64 = np.int64
uint64 = np.uint64
uint8 = np.uint8
bool_ = np.bool_
str_ = "U32"

# MC/DC types, will be defined by input deck
particle = None
particle_record = None
nuclide = None
material = None
surface = None
cell = None
universe = None
lattice = None
source = None
tally = None
technique = None
translate = None
group_array = None
j_array = None
global_ = None



# ==============================================================================
# Alignment Logic
# ==============================================================================


def fixup_dims(dim_tuple):
    return tuple([max(d,1) for d in dim_tuple])



def align(field_list):
    result = []
    offset = 0
    pad_id = 0
    for field in field_list:
        if len(field) > 3:
            print_error("Unexpected struct field specification. Specifications \
                        usually only consist of 3 or fewer members")
        multiplier = 1
        if len(field) == 3:
            field = ( field[0], field[1], fixup_dims(field[2]) )
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

        if multiplier == 0:
            multiplier = 1
        
        size *= multiplier

        if offset % alignment != 0:
            pad_size = alignment - (offset%alignment)
            result.append((f"padding_{pad_id}", uint8, (pad_size,)))
            pad_id += 1
            offset += pad_size

        result.append(field)
        offset += size
    
    if offset % 8 != 0:
        pad_size = 8 - (offset%8)
        result.append((f"padding_{pad_id}", uint8, (pad_size,)))
        pad_id += 1

    print(result)

    return result

        


def into_dtype(field_list):
    return np.dtype(align(field_list),align)
         


# ==============================================================================
# Copy Logic
# ==============================================================================


type_roster = {}


def copy_fn_for(kind,name):
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
    return eval(f'copy_{name}')






# ==============================================================================
# Particle
# ==============================================================================





# Particle (in-flight)
def make_type_particle(iQMC, G, tracked):
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
    # iqmc vector of weights
    Ng = 1
    if iQMC:
        Ng = G
    struct += [("iqmc_w", float64, (Ng,))]
    if tracked:
        struct += [("track_pid",int64),("track_hid",int64)]
    particle = into_dtype(struct)


# Particle record (in-bank)
def make_type_particle_record(iQMC, G, tracked):
    global particle_record
    struct = [
        ("x", float64),
        ("y", float64),
        ("z", float64),
        ("t", float64),
        ("ux", float64),
        ("uy", float64),
        ("uz", float64),
        ("g", uint64),
        ("w", float64),
        ("sensitivity_ID", int64),
        ("rng_seed", uint64),
    ]
    # iqmc vector of weights
    Ng = 1
    if iQMC:
        Ng = G
    if tracked:
        struct += [("track_pid",int64),("track_hid",int64)]
    struct += [("iqmc_w", float64, (Ng,))]
    particle_record = into_dtype(struct)


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
        [("particles", particle_record, (max_size,)), ("size", int64, (1,)), ("tag", "U16")]
    )


def precursor_bank(max_size):
    return into_dtype(
        [("precursors", precursor, (max_size,)), ("size", int64, (1,)), ("tag", "U16")]
    )


# ==============================================================================
# Nuclide and Material
# ==============================================================================


def make_type_nuclide(G, J):
    global nuclide
    nuclide = into_dtype(
        [
            ("ID", int64),
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
            ("sensitivity", bool_),
            ("sensitivity_ID", int64),
            ("dsm_Np", float64),
        ]
    )


def make_type_material(G, J, Nmax_nuclide):
    global material
    material = into_dtype(
        [
            ("ID", int64),
            ("N_nuclide", int64),
            ("nuclide_IDs", int64, (Nmax_nuclide,)),
            ("nuclide_densities", float64, (Nmax_nuclide,)),
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
            ("sensitivity", bool_),
        ]
    )


# ==============================================================================
# Surface
# ==============================================================================


def make_type_surface(Nmax_slice):
    global surface

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


def make_type_cell(Nmax_surface):
    global cell

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


def make_type_universe(Nmax_cell):
    global universe

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


def make_type_lattice(cards):
    global lattice

    # Max dimensional grids
    Nmax_x = 0
    Nmax_y = 0
    Nmax_z = 0
    for card in cards:
        Nmax_x = max(Nmax_x, card["mesh"]["Nx"])
        Nmax_y = max(Nmax_y, card["mesh"]["Ny"])
        Nmax_z = max(Nmax_z, card["mesh"]["Nz"])

    lattice = into_dtype(
        [("mesh", mesh_uniform), ("universe_IDs", int64, (Nmax_x, Nmax_y, Nmax_z))]
    )


# ==============================================================================
# Source
# ==============================================================================


def make_type_source(G):
    global source
    source = into_dtype(
        [
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
            ("group", float64, (G,)),
            ("time", float64, (2,)),
            ("prob", float64),
        ]
    )


# ==============================================================================
# Tally
# ==============================================================================


# Score lists
score_tl_list = (
    "flux",
    "current",
    "eddington",
    "density",
    "fission",
    "total",
)
score_x_list = (
    "flux_x",
    "current_x",
    "eddington_x",
    "density_x",
    "fission_x",
    "total_x",
)
score_y_list = (
    "flux_y",
    "current_y",
    "eddington_y",
    "density_y",
    "fission_y",
    "total_y",
)
score_z_list = (
    "flux_z",
    "current_z",
    "eddington_z",
    "density_z",
    "fission_z",
    "total_z",
)
score_t_list = (
    "flux_t",
    "current_t",
    "eddington_t",
    "density_t",
    "fission_t",
    "total_t",
)

score_list = score_tl_list + score_x_list + score_y_list + score_z_list + score_t_list


def make_type_tally(Ns, card):
    global tally

    def make_type_score(shape):
        return into_dtype(
            [
                ("bin", float64, shape),
                ("mean", float64, shape),
                ("sdev", float64, shape),
            ]
        )

    # Estimator flags
    struct = [
        ("tracklength", bool_),
        ("crossing", bool_),
        ("crossing_x", bool_),
        ("crossing_y", bool_),
        ("crossing_z", bool_),
        ("crossing_t", bool_),
    ]

    # Mesh
    mesh, Nx, Ny, Nz, Nt, Nmu, N_azi, Ng = make_type_mesh(card["mesh"])
    struct += [("mesh", mesh)]

    # Scores and shapes
    scores_shapes = [
        ["flux", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["density", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["fission", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["total", (Ns, Ng, Nt, Nx, Ny, Nz, Nmu, N_azi)],
        ["flux_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, Nmu, N_azi)],
        ["density_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, Nmu, N_azi)],
        ["fission_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, Nmu, N_azi)],
        ["total_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, Nmu, N_azi)],
        ["flux_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, Nmu, N_azi)],
        ["density_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, Nmu, N_azi)],
        ["fission_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, Nmu, N_azi)],
        ["total_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, Nmu, N_azi)],
        ["flux_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, Nmu, N_azi)],
        ["density_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, Nmu, N_azi)],
        ["fission_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, Nmu, N_azi)],
        ["total_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, Nmu, N_azi)],
        ["flux_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, Nmu, N_azi)],
        ["density_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, Nmu, N_azi)],
        ["fission_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, Nmu, N_azi)],
        ["total_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, Nmu, N_azi)],
        ["current", (Ns, Ng, Nt, Nx, Ny, Nz, 3)],
        ["current_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, 3)],
        ["current_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, 3)],
        ["current_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, 3)],
        ["current_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, 3)],
        ["eddington", (Ns, Ng, Nt, Nx, Ny, Nz, 6)],
        ["eddington_x", (Ns, Ng, Nt, Nx + 1, Ny, Nz, 6)],
        ["eddington_y", (Ns, Ng, Nt, Nx, Ny + 1, Nz, 6)],
        ["eddington_z", (Ns, Ng, Nt, Nx, Ny, Nz + 1, 6)],
        ["eddington_t", (Ns, Ng, Nt + 1, Nx, Ny, Nz, 6)],
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


setting = into_dtype(
    [
        ("N_particle", int64),
        ("N_inactive", int64),
        ("N_active", int64),
        ("N_cycle", int64),
        ("rng_seed", uint64),
        ("rng_stride", uint64),
        ("rng_g", uint64),
        ("rng_c", uint64),
        ("rng_mod", uint64),
        ("time_boundary", float64),
        ("bank_active_buff", int64),
        ("mode_eigenvalue", bool_),
        ("k_init", float64),
        ("gyration_radius", bool_),
        ("gyration_radius_type", int64),
        ("output", str_),
        ("progress_bar", bool_),
        ("source_file", bool_),
        ("source_file_name", str_),
        ("track_particle", bool_),
        ("save_particle", bool_),
        ("save_input_deck", bool_),
        ("IC_file", bool_),
        ("IC_file_name", str_),
        ("N_precursor", int64),
        ("N_sensitivity", int64),
    ]
)


# ==============================================================================
# Technique
# ==============================================================================


def make_type_technique(N_particle, G, card):
    global technique

    # Technique flags
    struct = [
        ("weighted_emission", bool_),
        ("implicit_capture", bool_),
        ("population_control", bool_),
        ("weight_window", bool_),
        ("weight_roulette", bool_),
        ("iQMC", bool_),
        ("IC_generator", bool_),
        ("time_census", bool_),
        ("branchless_collision", bool_),
    ]

    # =========================================================================
    # Population control
    # =========================================================================

    struct += [("pct", int64), ("pc_factor", float64)]

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
    struct += [("wr_threshold", float64), ("wr_chance", float64)]

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

    # Mesh (for qmc source tallies)
    if card["iQMC"]:
        mesh, Nx, Ny, Nz, Nt, Nmu, N_azi = make_type_mesh_(card["iqmc_mesh"])
        Ng = G
        N_dim = 6  # group, x, y, z, mu, phi
    else:
        Nx = Ny = Nz = Nt = Nmu = N_azi = N_particle = Ng = N_dim = 0

    struct += [("iqmc_mesh", mesh)]
    # Low-discprenecy sequence
    # TODO: make N_dim an input setting
    struct += [("iqmc_lds", float64, (N_particle, N_dim))]

    # Source
    struct += [("iqmc_source", float64, (Ng, Nt, Nx, Ny, Nz))]
    struct += [("iqmc_fixed_source", float64, (Ng, Nt, Nx, Ny, Nz))]
    struct += [("iqmc_material_idx", int64, (Nt, Nx, Ny, Nz))]

    # flux tallies
    struct += [("iqmc_flux", float64, (Ng, Nt, Nx, Ny, Nz))]
    struct += [("iqmc_flux_old", float64, (Ng, Nt, Nx, Ny, Nz))]
    struct += [("iqmc_flux_outter", float64, (Ng, Nt, Nx, Ny, Nz))]
    # if card.setting["mode_eigenvalue"]:
    #     struct += [("iqmc_flux_outter", float64, (Ng, Nt, Nx, Ny, Nz))]
    # else:
    #     struct += [("iqmc_flux_outter", float64, (0, 0, 0, 0, 0))]

    # Constants
    struct += [
        ("iqmc_maxitt", int64),
        ("iqmc_tol", float64),
        ("iqmc_itt", int64),
        ("iqmc_itt_outter", int64),
        ("iqmc_res", float64),
        ("iqmc_res_outter", float64),
        ("iqmc_N_dim", int64),
        ("iqmc_scramble", bool_),
        ("iqmc_seed", int64),
        ("iqmc_generator", str_),
        ("iqmc_fixed_source_solver", str_),
        ("iqmc_eigenmode_solver", str_),
        ("iqmc_krylov_restart", int64),
        ("iqmc_preconditioner_sweeps", int64),
        ("iqmc_sweep_counter", int64),
    ]

    # =========================================================================
    # Time census
    # =========================================================================

    struct += [
        ("census_time", float64, (len(card["census_time"]),)),
        ("census_idx", int64),
    ]

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
        ("IC_cycle_stretch", float64),
        ("IC_bank_neutron_local", bank_neutron_local),
        ("IC_bank_precursor_local", bank_precursor_local),
        ("IC_bank_neutron", bank_neutron),
        ("IC_bank_precursor", bank_precursor),
        ("IC_fission_score", float64, (1,) ),
        ("IC_fission", float64),
    ]

    # =========================================================================
    # Derivative Source Method
    # =========================================================================

    struct += [
        ("dsm_order", int64),
    ]

    # Finalize technique type
    technique = into_dtype(struct)


# ==============================================================================
# Global
# ==============================================================================

def copy_global(dst,src):
    pass

def make_type_global(card):
    global global_

    # Some numbers
    N_nuclide = len(card.nuclides)
    N_material = len(card.materials)
    N_surface = len(card.surfaces)
    N_cell = len(card.cells)
    N_source = len(card.sources)
    N_universe = len(card.universes)
    N_lattice = len(card.lattices)
    N_particle = card.setting["N_particle"]
    N_precursor = card.setting["N_precursor"]
    N_cycle = card.setting["N_cycle"]
    bank_active_buff = card.setting["bank_active_buff"]
    bank_census_buff = card.setting["bank_census_buff"]
    J = card.materials[0]["J"]
    N_work = math.ceil(N_particle / MPI.COMM_WORLD.Get_size())
    N_work_precursor = math.ceil(N_precursor / MPI.COMM_WORLD.Get_size())

    # Particle bank types
    bank_active = particle_bank(1 + bank_active_buff)
    if card.setting["mode_eigenvalue"] or card.technique["time_census"]:
        bank_census = particle_bank(int((1 + bank_census_buff) * N_work))
        bank_source = particle_bank(int((1 + bank_census_buff) * N_work))
    else:
        bank_census = particle_bank(0)
        bank_source = particle_bank(0)
    bank_precursor = precursor_bank(0)

    # Particle tracker
    N_track = 0
    if card.setting["track_particle"]:
        N_track = N_work * 1000

    # iQMC bank adjustment
    if card.technique["iQMC"]:
        bank_source = particle_bank(N_work)
        if card.setting["mode_eigenvalue"]:
            bank_census = particle_bank(0)

    # Source and IC files bank adjustments
    if not card.setting["mode_eigenvalue"]:
        if card.setting["source_file"]:
            bank_source = particle_bank(N_work)
        if card.setting["IC_file"]:
            bank_source = particle_bank(N_work)
            bank_precursor = precursor_bank(N_precursor)

    if (
        card.setting["source_file"] and not card.setting["mode_eigenvalue"]
    ) or card.technique["iQMC"]:
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
            ("bank_active", bank_active),
            ("bank_census", bank_census),
            ("bank_source", bank_source),
            ("bank_precursor", bank_precursor),
            ("rng_seed_base", uint64),
            ("rng_seed", uint64),
            ("source_seed", uint64),
            ("source_precursor_seed", uint64),
            ("rng_stride", int64),
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
            ("i_cycle", int64),
            ("cycle_active", bool_),
            ("eigenvalue_tally_nuSigmaF", float64, (1,)),
            ("eigenvalue_tally_n", float64, (1,)),
            ("eigenvalue_tally_C", float64, (1,)),
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
        ]
    )



# ==============================================================================
# Util
# ==============================================================================

def make_type_translate():
    global translate
    translate = into_dtype([("values", float64, (3,))])



def make_type_group_array(G):
    global group_array
    group_array = into_dtype([("values", float64, (G,))])


def make_type_j_array(J):
    global j_array
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
