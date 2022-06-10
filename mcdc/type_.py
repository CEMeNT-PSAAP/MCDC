import numpy as np
import numba as nb
import sys

float64 = np.float64
int64   = np.int64
uint64  = np.uint64
bool_   = np.bool_

# ==============================================================================
# Particle
# ==============================================================================

particle = np.dtype([('x', float64), ('y', float64), ('z', float64),
                     ('ux', float64), ('uy', float64), ('uz', float64),
                     ('time', float64), ('speed', float64), ('group', uint64),
                     ('weight', float64), ('alive', bool_), ('cell_ID', uint64), 
                     ('surface_ID', int64)])

@nb.njit
def make_particle():
    P = np.zeros(1, dtype=particle)[0]
    P['x']          = 0.0
    P['y']          = 0.0
    P['z']          = 0.0
    P['ux']         = 0.0
    P['uy']         = 0.0
    P['uz']         = 0.0
    P['group']      = 0
    P['time']       = 0.0
    P['weight']     = 1.0
    P['alive']      = True
    P['speed']      = 1.0
    P['cell_ID']    = -1
    P['surface_ID'] = -1
    return P

@nb.jit
def copy(P):
    P_new = np.zeros(1, dtype=particle)[0]
    P_new['x']          = P['x']         
    P_new['y']          = P['y']         
    P_new['z']          = P['z']         
    P_new['ux']         = P['ux']        
    P_new['uy']         = P['uy']        
    P_new['uz']         = P['uz']        
    P_new['group']      = P['group']     
    P_new['time']       = P['time']      
    P_new['weight']     = P['weight']    
    P_new['alive']      = P['alive']     
    P_new['speed']      = P['speed']     
    P_new['cell_ID']    = P['cell_ID']   
    P_new['surface_ID'] = P['surface_ID']
    return P_new


# ==============================================================================
# Particle bank
# ==============================================================================

def particle_bank(max_size):
    return np.dtype([('particles', particle, (max_size,)), ('size', int64)])

# ==============================================================================
# Material
# ==============================================================================

material = None
def make_type_material(G,J):
    global material
    material = np.dtype([('ID', uint64),
                         ('G', uint64), ('J', uint64),
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

surface = np.dtype([('ID', uint64), 
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
def make_type_cell(N_surfaces):
    global cell

    cell = np.dtype([('ID', uint64),
                     ('N_surfaces', uint64),
                     ('surface_IDs', uint64, (N_surfaces,)),
                     ('positive_flags', bool_, (N_surfaces,)),
                     ('material_ID', uint64)])

# ==============================================================================
# Source
# ==============================================================================

source = None
def make_type_source(G):
    global source
    source = np.dtype([('ID', uint64),
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
score_list = ('flux', 'flux_x', 'flux_t', 
              'current', 'current_x', 'current_t',
              'eddington', 'eddington_x', 'eddington_t')

def make_type_mesh(card):
    Nx = len(card['x']) - 1
    Ny = len(card['y']) - 1
    Nz = len(card['z']) - 1
    Nt = len(card['t']) - 1
    return  np.dtype([('x', float64, (Nx+1,)), ('y', float64, (Ny+1,)),
                      ('z', float64, (Nz+1,)), ('t', float64, (Nt+1,))]),\
            Nx, Ny, Nz, Nt

def make_type_tally(card, Ng, N_iter):
    global tally

    # Estimator flags
    struct = [('tracklength', bool_), ('crossing', bool_), 
              ('crossing_x', bool_), ('crossing_t', bool_)]

    # Mesh
    mesh, Nx, Ny, Nz, Nt = make_type_mesh(card['mesh'])
    struct += [('mesh', mesh)]

    # Score type generator
    def score(shape):
        return np.dtype([('bin', float64, shape),
                         ('sum', float64, shape),
                         ('sum_sq', float64, shape),
                         ('mean', float64, (N_iter,*shape)), 
                         ('sdev', float64, (N_iter,*shape))])

    # Scores and shapes
    scores_shapes = [
                     ['flux', (Ng, Nt, Nx, Ny, Nz)],
                     ['flux_x', (Ng, Nt, Nx+1, Ny, Nz)],
                     ['flux_t', (Ng, Nt+1, Nx, Ny, Nz)],
                     ['current', (Ng, Nt, Nx, Ny, Nz, 3)],
                     ['current_x', (Ng, Nt, Nx+1, Ny, Nz, 3)],
                     ['current_t', (Ng, Nt+1, Nx, Ny, Nz, 3)],
                     ['eddington', (Ng, Nt, Nx, Ny, Nz, 6)],
                     ['eddington_x', (Ng, Nt, Nx+1, Ny, Nz, 6)],
                     ['eddington_t', (Ng, Nt+1, Nx, Ny, Nz, 6)]
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
        if not card[name]:
            shape = (0,)*len(shape)
        scores_struct += [(name, score(shape))]
    scores = np.dtype(scores_struct)
    struct += [('score', scores)]
   
    # Make tally structure
    tally = np.dtype(struct)

# ==============================================================================
# Setting
# ==============================================================================

setting = np.dtype([('N_hist', uint64), ('N_iter', uint64),
                    ('mode_eigenvalue', bool_), ('mode_alpha', bool_),
                    ('time_boundary', float64),
                    ('rng_seed', int64), ('rng_stride', int64),
                    ('k_init', float64), ('alpha_init', float64),
                    ('output_name', 'U10'), ('progress_bar', bool_)])

# ==============================================================================
# Technique
# ==============================================================================

technique = None

def make_type_technique(card):
    global technique

    struct = [('weighted_emission', bool_), ('implicit_capture', bool_),
              ('population_control', bool_), ('weight_window', bool_)]

    # Weight window
    # Mesh
    mesh, Nx, Ny, Nz, Nt = make_type_mesh(card['ww_mesh'])
    struct += [('ww_mesh', mesh)]
    # Window
    struct += [('ww', float64, (Nt, Nx, Ny, Nz))]

    technique = np.dtype(struct)

# ==============================================================================
# Global
# ==============================================================================

global_ = None

def make_type_global(card):
    global global_

    # Some numbers
    N_materials = len(card.materials)
    N_surfaces  = len(card.surfaces)
    N_cells     = len(card.cells)
    N_sources   = len(card.sources)
    N_hist      = card.setting['N_hist']
    N_iter      = card.setting['N_iter']
    if not card.setting['mode_eigenvalue']: N_iter = 0

    # Particle bank types
    bank_history = particle_bank(100)
    if card.setting['mode_eigenvalue']:
        bank_census  = particle_bank(5*N_hist)
        bank_source  = particle_bank(5*N_hist)
    else:
        bank_census  = particle_bank(0)
        bank_source  = particle_bank(0)

    global_ = np.dtype([('materials', material, (N_materials,)),
                        ('surfaces', surface, (N_surfaces,)),
                        ('cells', cell, (N_cells,)),
                        ('sources', source, (N_sources,)),
                        ('tally', tally),
                        ('setting', setting),
                        ('technique', technique),
                        ('bank_history', bank_history),
                        ('bank_census', bank_census),
                        ('bank_source', bank_source),
                        ('rng_seed_base', int64),
                        ('rng_seed', int64),
                        ('rng_g', int64),
                        ('rng_c', int64),
                        ('rng_mod', uint64),
                        ('rng_stride', int64),
                        ('k_eff', float64),
                        ('k_iterate', float64, (N_iter,)),
                        ('alpha_eff', float64),
                        ('alpha_iterate', float64, (N_iter,)),
                        ('nuSigmaF', float64),
                        ('inverse_speed', float64),
                        ('runtime_total', float64),
                        ('i_iter', int64),
                        ('mpi_size', uint64),
                        ('mpi_rank', uint64),
                        ('mpi_work_start', uint64),
                        ('mpi_work_size', uint64),
                        ('mpi_work_size_total', uint64)])
