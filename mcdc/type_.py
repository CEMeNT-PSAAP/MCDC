import math
import numpy as np
import sys

from mpi4py import MPI

float64 = np.float64
int64   = np.int64
uint64  = np.uint64
bool_   = np.bool_

# ==============================================================================
# Particle
# ==============================================================================

# Particle (in-flight)
particle = np.dtype([
    ('x', float64), ('y', float64), ('z', float64), ('t', float64),
    ('ux', float64), ('uy', float64), ('uz', float64), ('g', uint64), 
    ('w', float64),
    ('material_ID', int64), ('cell_ID', int64), 
    ('surface_ID', int64), ('translation', float64, (3,)),
    ('event', int64)])

# Particle record (in-bank)
particle_record = np.dtype([
    ('x', float64), ('y', float64), ('z', float64), ('t', float64),
    ('ux', float64), ('uy', float64), ('uz', float64), ('g', uint64), 
    ('w', float64)])

# Records (for IC generator, )
neutron   = np.dtype([('x', float64), ('y', float64), ('z', float64),
                      ('ux', float64), ('uy', float64), ('uz', float64),
                      ('g', uint64), ('w', float64)])
precursor = np.dtype([('x', float64), ('y', float64), ('z', float64),
                      ('g', uint64), ('w', float64)])

# ==============================================================================
# Particle bank
# ==============================================================================

def particle_bank(max_size):
    return np.dtype([('particles', particle_record, (max_size,)), 
                     ('size', int64)])

# ==============================================================================
# Material
# ==============================================================================

material = None
def make_type_material(G,J):
    global material
    material = np.dtype([('ID', int64),
                         ('G', int64), ('J', int64),
                         ('speed', float64, (G,)), ('decay', float64, (J,)),
                         ('total', float64, (G,)), ('capture', float64, (G,)),
                         ('scatter', float64, (G,)), ('fission', float64, (G,)),
                         ('nu_s', float64, (G,)), ('nu_f', float64, (G,)),
                         ('nu_p', float64, (G,)), ('nu_d', float64, (G,J)), 
                         ('chi_s', float64, (G,G)), ('chi_p', float64, (G,G)),
                         ('chi_d', float64, (J,G))])

# ==============================================================================
# Surface
# ==============================================================================

surface = np.dtype([('ID', int64), 
                    ('vacuum', bool_), ('reflective', bool_),
                    ('A', float64), ('B', float64), ('C', float64), 
                    ('D', float64), ('E', float64), ('F', float64), 
                    ('G', float64), ('H', float64), ('I', float64), 
                    ('J', float64), ('linear', bool_), 
                    ('nx', float64), ('ny', float64), ('nz', float64)])

# ==============================================================================
# Cell
# ==============================================================================

cell = None
def make_type_cell(Nmax_surface):
    global cell

    cell = np.dtype([('ID', int64),
                     ('N_surface', int64),
                     ('surface_IDs', int64, (Nmax_surface,)),
                     ('positive_flags', bool_, (Nmax_surface,)),
                     ('material_ID', int64),
                     ('lattice', bool_), ('lattice_ID', int64),
                     ('lattice_center', float64, (3,))])

# ==============================================================================
# Universe
# ==============================================================================

universe = None
def make_type_universe(Nmax_cell):
    global universe

    universe = np.dtype([('ID', int64),
                         ('N_cell', int64),
                         ('cell_IDs', int64, (Nmax_cell,))])

# ==============================================================================
# Lattice
# ==============================================================================

mesh_uniform = np.dtype([('x0', float64), ('dx', float64), ('Nx', int64),
                         ('y0', float64), ('dy', float64), ('Ny', int64),
                         ('z0', float64), ('dz', float64), ('Nz', int64)])

lattice = None
def make_type_lattice(cards):
    global lattice

    # Max dimensional grids
    Nmax_x = 0
    Nmax_y = 0
    Nmax_z = 0
    for card in cards:
        Nmax_x = max(Nmax_x, card['mesh']['Nx'])
        Nmax_y = max(Nmax_y, card['mesh']['Ny'])
        Nmax_z = max(Nmax_z, card['mesh']['Nz'])

    lattice = np.dtype([('mesh', mesh_uniform),
                        ('universe_IDs', int64, (Nmax_x, Nmax_y, Nmax_z))])

# ==============================================================================
# Source
# ==============================================================================

source = None
def make_type_source(G):
    global source
    source = np.dtype([('ID', int64),
                       ('box', bool_), ('isotropic', bool_),
                       ('x', float64), ('y', float64), ('z', float64),
                       ('box_x', float64, (2,)), ('box_y', float64, (2,)), 
                       ('box_z', float64, (2,)),
                       ('ux', float64), ('uy', float64), ('uz', float64),
                       ('group', float64, (G,)), ('time', float64, (2,)),
                       ('prob', float64)])

# ==============================================================================
# Tally
# ==============================================================================

tally = None

# Score lists
score_tl_list = ('flux',   'current',   'eddington',   'density',   'fission',   'total')
score_x_list  = ('flux_x', 'current_x', 'eddington_x', 'density_x', 'fission_x', 'total_x')
score_y_list  = ('flux_y', 'current_y', 'eddington_y', 'density_y', 'fission_y', 'total_y')
score_z_list  = ('flux_z', 'current_z', 'eddington_z', 'density_z', 'fission_z', 'total_z')
score_t_list  = ('flux_t', 'current_t', 'eddington_t', 'density_t', 'fission_t', 'total_t')

score_list = score_tl_list + score_x_list + score_y_list + score_z_list\
                           + score_t_list

def make_type_tally(card):
    global tally
 
    def make_type_score(shape):
        return np.dtype([('bin', float64, shape),
                         ('mean', float64, shape),
                         ('sdev', float64, shape)])

    # Estimator flags
    struct = [('tracklength', bool_), ('crossing', bool_), 
              ('crossing_x', bool_), ('crossing_y', bool_), 
              ('crossing_z', bool_), ('crossing_t', bool_)]

    # Mesh
    mesh, Nx, Ny, Nz, Nt = make_type_mesh(card.tally['mesh'])
    struct += [('mesh', mesh)]

    # Scores and shapes
    Ng = card.materials[0]['G']
    scores_shapes = [
                     ['flux',        (Ng, Nt, Nx, Ny, Nz)],
                     ['density',     (Ng, Nt, Nx, Ny, Nz)],
                     ['fission',     (Ng, Nt, Nx, Ny, Nz)],
                     ['total',     (Ng, Nt, Nx, Ny, Nz)],
                     ['current',     (Ng, Nt, Nx, Ny, Nz, 3)],
                     ['eddington',   (Ng, Nt, Nx, Ny, Nz, 6)],
                     ['flux_x',      (Ng, Nt, Nx+1, Ny, Nz)],
                     ['density_x',   (Ng, Nt, Nx+1, Ny, Nz)],
                     ['fission_x',   (Ng, Nt, Nx+1, Ny, Nz)],
                     ['total_x',   (Ng, Nt, Nx+1, Ny, Nz)],
                     ['current_x',   (Ng, Nt, Nx+1, Ny, Nz, 3)],
                     ['eddington_x', (Ng, Nt, Nx+1, Ny, Nz, 6)],
                     ['flux_y',      (Ng, Nt, Nx, Ny+1, Nz)],
                     ['density_y',   (Ng, Nt, Nx, Ny+1, Nz)],
                     ['fission_y',   (Ng, Nt, Nx, Ny+1, Nz)],
                     ['total_y',   (Ng, Nt, Nx, Ny+1, Nz)],
                     ['current_y',   (Ng, Nt, Nx, Ny+1, Nz, 3)],
                     ['eddington_y', (Ng, Nt, Nx, Ny+1, Nz, 6)],
                     ['flux_z',      (Ng, Nt, Nx, Ny, Nz+1)],
                     ['density_z',   (Ng, Nt, Nx, Ny, Nz+1)],
                     ['fission_z',   (Ng, Nt, Nx, Ny, Nz+1)],
                     ['total_z',   (Ng, Nt, Nx, Ny, Nz+1)],
                     ['current_z',   (Ng, Nt, Nx, Ny, Nz+1, 3)],
                     ['eddington_z', (Ng, Nt, Nx, Ny, Nz+1, 6)],
                     ['flux_t',      (Ng, Nt+1, Nx, Ny, Nz)],
                     ['density_t',   (Ng, Nt+1, Nx, Ny, Nz)],
                     ['fission_t',   (Ng, Nt+1, Nx, Ny, Nz)],
                     ['total_t',   (Ng, Nt+1, Nx, Ny, Nz)],
                     ['current_t',   (Ng, Nt+1, Nx, Ny, Nz, 3)],
                     ['eddington_t', (Ng, Nt+1, Nx, Ny, Nz, 6)],
                    ]

    # Add score flags to structure
    for i in range(len(scores_shapes)):
        name    = scores_shapes[i][0]
        struct += [(name, bool_)]

    # Add scores to structure
    scores_struct = []
    for i in range(len(scores_shapes)):
        name  = scores_shapes[i][0]
        shape = scores_shapes[i][1]
        if not card.tally[name]:
            shape = (0,)*len(shape)
        scores_struct += [(name, make_type_score(shape))]
    scores = np.dtype(scores_struct)
    struct += [('score', scores)]
   
    # Make tally structure
    tally = np.dtype(struct)

# ==============================================================================
# Setting
# ==============================================================================

setting = np.dtype([('N_particle', int64), ('N_inactive', int64),
                    ('N_active', int64), ('N_cycle', int64),
                    ('rng_seed', int64), ('rng_stride', int64),
                    ('rng_g', int64), ('rng_c', int64), ('rng_mod', uint64),
                    ('time_boundary', float64), ('bank_max', int64),
                    ('mode_eigenvalue', bool_), ('k_init', float64),
                    ('gyration_radius', bool_), ('gyration_radius_type', int64),
                    ('output', 'U30'), ('progress_bar', bool_),
                    ('filed_source', bool_), ('source_file', 'U20')])

# ==============================================================================
# Technique
# ==============================================================================

technique = None

def make_type_technique(card):
    global technique

    # Technique flags
    struct = [('weighted_emission', bool_), ('implicit_capture', bool_),
              ('population_control', bool_), ('branchless_collision', bool_),
              ('weight_window', bool_), ('IC_generator', bool_)]

    # =========================================================================
    # Weight window
    # =========================================================================

    # Mesh
    mesh, Nx, Ny, Nz, Nt = make_type_mesh(card.technique['ww_mesh'])
    struct += [('ww_mesh', mesh)]
    
    # Window
    struct += [('ww', float64, (Nt, Nx, Ny, Nz))]

    # =========================================================================
    # IC generator
    # =========================================================================
   
    # Banks
    #   We need local banks to ensure reproducibility regardless of # of MPIs
    if card.technique['IC_generator']:
        Nn = int(card.technique['IC_Nn']*1.2)
        Np = int(card.technique['IC_Np']*1.2)
        Nn_local = Nn#int(Nn/card.setting['N_active'])
        Np_local = Np#int(Np/card.setting['N_active'])
    else:
        Nn = 0; Np = 0; Nn_local = 0; Np_local = 0
    bank_neutron   = np.dtype([('content', neutron, (Nn,)), ('size', int64)])
    bank_precursor = np.dtype([('content', precursor, (Np,)), ('size', int64)])
    bank_neutron_local   = np.dtype([('content', neutron, (Nn_local,)), ('size', int64)])
    bank_precursor_local = np.dtype([('content', precursor, (Np_local,)), ('size', int64)])

    struct += [('IC_Nn', int64), ('IC_Np', int64),
               ('IC_tally_n', float64), ('IC_tally_C', float64),
               ('IC_n_eff', float64), ('IC_C_eff', float64),
               ('IC_bank_neutron_local', bank_neutron_local), 
               ('IC_bank_precursor_local', bank_precursor_local),
               ('IC_bank_neutron', bank_neutron), 
               ('IC_bank_precursor', bank_precursor)]

    # Finalize technique type
    technique = np.dtype(struct)

# ==============================================================================
# Global
# ==============================================================================

global_ = None

def make_type_global(card):
    global global_

    # Some numbers
    N_material = len(card.materials)
    N_surface  = len(card.surfaces)
    N_cell     = len(card.cells)
    N_source   = len(card.sources)
    N_universe = len(card.universes)
    N_lattice  = len(card.lattices)
    N_particle = card.setting['N_particle']
    N_cycle    = card.setting['N_cycle']
    N_bank_max = card.setting['bank_max']
    J          = card.materials[0]['J']
    N_work     = math.ceil(N_particle/MPI.COMM_WORLD.Get_size())

    # Particle bank types
    bank_active = particle_bank(N_bank_max)
    if card.setting['mode_eigenvalue']:
        bank_census  = particle_bank(2*N_work)
        bank_source  = particle_bank(2*N_work)
    else:
        bank_census = particle_bank(0)
        bank_source = particle_bank(0)
    if card.setting['filed_source']:
        bank_source = particle_bank(N_work)

    # GLobal type
    global_ = np.dtype([('materials', material, (N_material,)),
                        ('surfaces', surface, (N_surface,)),
                        ('cells', cell, (N_cell,)),
                        ('universes', universe, (N_universe,)),
                        ('lattices', lattice, (N_lattice,)),
                        ('sources', source, (N_source,)),
                        ('tally', tally),
                        ('setting', setting),
                        ('technique', technique),
                        ('bank_active', bank_active),
                        ('bank_census', bank_census),
                        ('bank_source', bank_source),
                        ('rng_seed_base', int64),
                        ('rng_seed', int64),
                        ('rng_stride', int64),
                        ('k_eff', float64),
                        ('k_cycle', float64, (N_cycle,)),
                        ('k_avg', float64),
                        ('k_sdv', float64),
                        ('k_avg_running', float64),
                        ('k_sdv_running', float64),
                        ('gyration_radius', float64, (N_cycle,)),
                        ('i_cycle', int64),
                        ('cycle_active', bool_),
                        ('global_tally_nuSigmaF', float64),
                        ('mpi_size', int64),
                        ('mpi_rank', int64),
                        ('mpi_master', bool_),
                        ('mpi_work_start', int64),
                        ('mpi_work_size', int64),
                        ('mpi_work_size_total', int64),
                        ('runtime_total', float64)])

# ==============================================================================
# Util
# ==============================================================================

def make_type_mesh(card):
    Nx = len(card['x']) - 1
    Ny = len(card['y']) - 1
    Nz = len(card['z']) - 1
    Nt = len(card['t']) - 1
    return  np.dtype([('x', float64, (Nx+1,)), ('y', float64, (Ny+1,)),
                      ('z', float64, (Nz+1,)), ('t', float64, (Nt+1,))]),\
            Nx, Ny, Nz, Nt
