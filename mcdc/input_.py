import numpy as np

from mcdc.class_   import SurfaceHandle
from mcdc.constant import *
from mcdc.print_   import print_error

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

# ==============================================================================
# Material
# ==============================================================================

def material(capture=None, scatter=None, fission=None, nu_p=None, nu_d=None, 
             chi_p=None, chi_d=None, nu_s=None, speed=None, decay=None):
    """
    Arguments
    ---------
    capture : numpy.ndarray (1D)
        Capture cross-section [/cm]
    scatter : numpy.ndarray (2D)
        Differential scattering cross-section [gout][gin] [/cm].
    fission : numpy.ndarray (1D)
        Fission cross-section [/cm]. 
    *At least capture, scatter, or fission cross-section needs to be 
    provided.

    nu_s : numpy.ndarray (1D)
        Scattering multiplication.
    nu_p : numpy.ndarray (1D)
        Prompt fission neutron yield.
    nu_d : numpy.ndarray (2D)
        Delayed neutron precursor yield [dg][gin].
    *nu_p or nu_d is needed if fission is provided.
    
    chi_p : numpy.ndarray (2D)
        Prompt fission spectrum [gout][gin]
    chi_d : numpy.ndarray (2D)
        Delayed neutron spectrum [gout][dg]
    *chi_p and chi_d are needed if nu_p and nu_d are provided, respectively.

    speed : numpy.ndarray (1D)
        Energy group speed
    decay : numpy.ndarray (1D)
        Precursor group decay constant [/s]
    *speed and decay are optional. By default, values for speed and decay 
    are one and infinite, respectively. Universal speed and decay can be 
    provided through mcdc.set_universal_speed and mcdc.set_universal_decay.
    """

    # Energy group size
    if capture is not None:
        G = len(capture)
    elif scatter is not None:
        G = len(scatter)
    elif fission is not None:
        G = len(fission)
    else:
        print_error("Need to supply capture, scatter, or fission to "\
                    + "mcdc.material")

    # Delayed group size
    J = 0
    if nu_d is not None:
        J = len(nu_d)
    
    # Set default card values (c.f. type_.py)
    card            = {}
    card['tag']     = 'Material'
    card['ID']      = len(mcdc.input_card.materials)
    card['G']       = G
    card['J']       = J
    card['speed']   = np.ones(G)
    card['decay']   = np.ones(J)*INF
    card['capture'] = np.zeros(G)
    card['scatter'] = np.zeros(G)
    card['fission'] = np.zeros(G)
    card['total']   = np.zeros(G)
    card['nu_s']    = np.ones(G)
    card['nu_p']    = np.zeros(G)
    card['nu_d']    = np.zeros([G,J])
    card['nu_f']    = np.zeros(G)
    card['chi_s']   = np.zeros([G,G])
    card['chi_p']   = np.zeros([G,G])
    card['chi_d']   = np.zeros([J,G])

    # Speed
    if speed is not None:
        card['speed'][:] = speed[:]

    # Decay constant
    if decay is not None:
        card['decay'][:] = decay[:]

    # Cross-sections
    if capture is not None:
        card['capture'][:] = capture[:]
    if scatter is not None:
        card['scatter'][:] = np.sum(scatter,0)[:]
    if fission is not None:
        card['fission'][:] = fission[:]
    card['total'][:] = card['capture'] + card['scatter'] + card['fission']
    
    # Scattering multiplication
    if nu_s is not None:
        card['nu_s'][:,:] = nu_s[:,:]

    # Fission productions
    if fission is not None:
        if nu_p is None and nu_d is None:
            print_error("Need to supply nu_p or nu_d for fissionable "\
                        + "mcdc.material")
    if nu_p is not None:
        card['nu_p'][:] = nu_p[:]
    if nu_d is not None:
        card['nu_d'][:,:] = np.swapaxes(nu_d, 0, 1)[:,:] # [dg,gin] -> [gin,dg]
    card['nu_f'] += card['nu_p']
    for j in range(J):
        card['nu_f'] += card['nu_d'][:,j]

    # Scattering spectrum
    if scatter is not None:
        card['chi_s'][:,:] = np.swapaxes(scatter, 0, 1)[:,:] # [gout,gin] -> [gin,gout]
        for g in range(G): 
            if card['scatter'][g] > 0.0:
                card['chi_s'][g,:] /= card['scatter'][g]

    # Fission spectrums
    if nu_p is not None:
        if G == 1:
            card['chi_p'][:,:] = np.array([[1.0]])
        elif chi_p is None:
            print_error("Need to supply chi_p if nu_p is provided")
        else:
            card['chi_p'][:,:] = np.swapaxes(chi_p, 0, 1)[:,:] # [gout,gin] -> [gin,gout]
            # Normalize
            for g in range(G):
                if np.sum(card['chi_p'][g,:]) > 0.0:
                    card['chi_p'][g,:] /= np.sum(card['chi_p'][g,:])
    if nu_d is not None:
        if G == 1:
            card['chi_d'][:,:] = np.ones([J,G])
        else:
            if chi_d is None:
                print_error("Need to supply chi_d if nu_d is provided")
            card['chi_d'][:,:] = np.swapaxes(chi_d, 0, 1)[:,:] # [gout,dg] -> [dg,gout]
        # Normalize
        for dg in range(J):
            if np.sum(card['chi_d'][dg,:]) > 0.0:
                card['chi_d'][dg,:] /= np.sum(card['chi_d'][dg,:])

    # Push card
    mcdc.input_card.materials.append(card)
    return card

# ==============================================================================
# Surface
# ==============================================================================

def surface(type_, **kw):
    # Set default card values (c.f. type_.py)
    card               = {}
    card['tag']        = 'Surface'
    card['ID']         = len(mcdc.input_card.surfaces)
    card['vacuum']     = False
    card['reflective'] = False
    card['A']          = 0.0
    card['B']          = 0.0
    card['C']          = 0.0
    card['D']          = 0.0
    card['E']          = 0.0
    card['F']          = 0.0
    card['G']          = 0.0
    card['H']          = 0.0
    card['I']          = 0.0
    card['J']          = 0.0
    card['linear']     = False
    card['nx']         = 0.0
    card['ny']         = 0.0
    card['nz']         = 0.0

    # Boundary condition
    bc = kw.get('bc')
    if bc is not None:
        if bc == 'vacuum':
            card['vacuum'] = True
        elif bc == 'reflective':
            card['reflective'] = True
        else:
            print_error("Unsupported surface boundary condition: "+bc)

    # Surface type
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    if type_ == 'plane-x':
        card['G'] = 1.0
        card['J'] = -kw.get('x')
        card['linear'] = True
    elif type_ == 'plane-y':
        card['H'] = 1.0
        card['J'] = -kw.get('y')
        card['linear'] = True
    elif type_ == 'plane-z':
        card['I'] = 1.0
        card['J'] = -kw.get('z')
        card['linear'] = True
    elif type_ == 'plane':
        card['G'] = kw.get('A')
        card['H'] = kw.get('B')
        card['I'] = kw.get('C')
        card['J'] = kw.get('D')
        card['linear'] = True
    elif type_ == 'cylinder-x':
        y, z = kw.get('center')[:]
        r    = kw.get('radius')
        card['B'] = 1.0
        card['C'] = 1.0
        card['H'] = -2.0*y
        card['I'] = -2.0*z
        card['J'] = y**2 + z**2 - r**2
    elif type_ == 'cylinder-y':
        x, z = kw.get('center')[:]
        r    = kw.get('radius')
        card['A'] = 1.0
        card['C'] = 1.0
        card['G'] = -2.0*x
        card['I'] = -2.0*z
        card['J'] = x**2 + z**2 - r**2
    elif type_ == 'cylinder-z':
        x, y = kw.get('center')[:]
        r    = kw.get('radius')
        card['A'] = 1.0
        card['B'] = 1.0
        card['G'] = -2.0*x
        card['H'] = -2.0*y
        card['J'] = x**2 + y**2 - r**2
    elif type_ == 'sphere':
        x, y, z = kw.get('center')[:]
        r       = kw.get('radius')
        card['A'] = 1.0
        card['B'] = 1.0
        card['C'] = 1.0
        card['G'] = -2.0*x
        card['H'] = -2.0*y
        card['I'] = -2.0*z
        card['J'] = x**2 + y**2 + z**2 - r**2
    elif type_ == 'quadric':
        card['A'] = kw.get('A')
        card['B'] = kw.get('B')
        card['C'] = kw.get('C')
        card['D'] = kw.get('D')
        card['E'] = kw.get('E')
        card['F'] = kw.get('F')
        card['G'] = kw.get('G')
        card['H'] = kw.get('H')
        card['I'] = kw.get('I')
        card['J'] = kw.get('J')
    else:
        print_error("Unsupported surface type: "+type_)

    # Normal vector if linear
    if card['linear']:
        nx = card['G']
        ny = card['H']
        nz = card['I']
        # Normalize
        norm = (nx**2 + ny**2 + nz**2)**0.5
        card['nx'] = nx/norm
        card['ny'] = ny/norm
        card['nz'] = nz/norm
    
    # Push card
    mcdc.input_card.surfaces.append(card)
    return SurfaceHandle(card)

# =============================================================================
# Cell
# =============================================================================

def cell(surfaces_flags, material):
    N_surfaces = len(surfaces_flags)

    # Set default card values (c.f. type_.py)
    card                   = {}
    card['tag']            = 'Cell'
    card['ID']             = len(mcdc.input_card.cells)
    card['N_surfaces']     = N_surfaces
    card['surface_IDs']    = np.zeros(N_surfaces, dtype=int)
    card['positive_flags'] = np.zeros(N_surfaces, dtype=bool)
    card['material_ID']    = 0

    # Surfaces and flags
    for i in range(N_surfaces):
        card['surface_IDs'][i]    = surfaces_flags[i][0]['ID']
        card['positive_flags'][i] = surfaces_flags[i][1]

    # Material
    card['material_ID'] = material['ID']
    
    # Push card
    mcdc.input_card.cells.append(card)
    return card

# =============================================================================
# Universe
# =============================================================================

def universe(cells):
    N_cells = len(cells)

    # Set default card values (c.f. type_.py)
    card             = {}
    card['tag']      = 'Universe'
    card['ID']       = len(mcdc.input_card.universes)
    card['N_cells']  = N_cells
    card['cell_IDs'] = np.zeros(N_cells, dtype=int)

    # Cells
    for i in range(N_cells):
        card['cell_IDs'][i] = cells[i]['ID']

    # Push card
    mcdc.input_card.universes.append(card)
    return card

#==============================================================================
# Lattice
#==============================================================================

def lattice(x=None, y=None, z=None, t=None, universes=None, 
            bc_x_plus=None, bc_x_minus=None, bc_y_plus=None, bc_y_minus=None,
            bc_z_plus=None, bc_z_minus=None):
    card = mcdc.input_card.lattice

    # Set mesh
    if x is not None: card['mesh']['x'] = x
    if y is not None: card['mesh']['y'] = y
    if z is not None: card['mesh']['z'] = z
    if t is not None: card['mesh']['t'] = t

    # Set universe IDs
    universe_IDs = np.array(universes, dtype=np.int64)
    ax_expand = []
    if t is None:
        ax_expand.append(0)
    if x is None:
        ax_expand.append(1)
    if y is None:
        ax_expand.append(2)
    if z is None:
        ax_expand.append(3)
    for ax in ax_expand:
        universe_IDs = np.expand_dims(universe_IDs, axis=ax)
    tmp = np.transpose(universe_IDs)
    tmp = np.flip(tmp, axis=2)
    card['universe_IDs'] = np.flip(tmp, axis=3)

    # Set BC
    if bc_x_plus == 'reflective':
        card['reflective_x+'] = True
    if bc_x_minus == 'reflective':
        card['reflective_x-'] = True
    if bc_y_plus == 'reflective':
        card['reflective_y+'] = True
    if bc_y_minus == 'reflective':
        card['reflective_y-'] = True
    if bc_z_plus == 'reflective':
        card['reflective_z+'] = True
    if bc_z_minus == 'reflective':
        card['reflective_z-'] = True

    return card

# ==============================================================================
# Source
# ==============================================================================

def source(**kw):
    # Get keyword arguments
    point     = kw.get('point')
    x         = kw.get('x')
    y         = kw.get('y')
    z         = kw.get('z')
    isotropic = kw.get('isotropic')
    direction = kw.get('direction')
    energy    = kw.get('energy')
    time      = kw.get('time')
    prob      = kw.get('prob')
        
    # Set default card values (c.f. type_.py)
    card              = {}
    card['tag']       = 'Source'
    card['ID']        = len(mcdc.input_card.sources)
    card['box']       = False
    card['isotropic'] = True
    card['x']         = 0.0
    card['y']         = 0.0
    card['z']         = 0.0
    card['box_x']     = np.array([0.0, 0.0])
    card['box_y']     = np.array([0.0, 0.0])
    card['box_z']     = np.array([0.0, 0.0])
    card['ux']        = 0.0
    card['uy']        = 0.0
    card['uz']        = 0.0
    card['group']     = np.array([1.0])
    card['time']      = np.array([0.0, 0.0])
    card['prob']      = 1.0
    
    # Set position
    if point is not None:
        card['x'] = point[0]
        card['y'] = point[1]
        card['z'] = point[2]
    else:
        card['box'] = True
        if x is not None:
            card['box_x'] = np.array(x)
        if y is not None:
            card['box_y'] = np.array(y)
        if z is not None:
            card['box_z'] = np.array(z)

    # Set direction
    if direction is not None:
        card['isotropic'] = False
        ux = direction[0]
        uy = direction[1]
        uz = direction[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2)**0.5
        card['ux'] = ux/norm
        card['uy'] = uy/norm
        card['uz'] = uz/norm

    # Set energy
    if energy is not None:
        group = np.array(energy)
        # Normalize
        card['group'] = group/np.sum(group)

    # Set time
    if time is not None:
        card['time'] = np.array(time)

    # Set probability
    if prob is not None:
        card['prob'] = prob

    # Push card
    mcdc.input_card.sources.append(card)
    return card

#==============================================================================
# Tally
#==============================================================================

def tally(scores, x=None, y=None, z=None, t=None):
    # Check if tally card has been initialized
    card = mcdc.input_card.tally

    # Set mesh
    if x is not None: card['mesh']['x'] = x
    if y is not None: card['mesh']['y'] = y
    if z is not None: card['mesh']['z'] = z
    if t is not None: card['mesh']['t'] = t

    # Set score flags
    for s in scores:
        if s == 'flux':
            card['flux'] = True
        elif s == 'current':
            card['current'] = True
        elif s == 'eddington':
            card['eddington'] = True
        elif s == 'flux-x':
            card['flux_x'] = True
        elif s == 'current-x':
            card['current_x'] = True
        elif s == 'flux-t':
            card['flux_t'] = True
        elif s == 'fission':
            card['fission'] = True
        elif s == 'density':
            card['density'] = True
        elif s == 'density-t':
            card['density_t'] = True
        else:
            print_error("Unknown tally score %s"%s)

    # Set estimator flags
    if card['flux'] or card['current'] or card['eddington'] or card['fission'] or card['density']:
        card['tracklength'] = True
    if card['flux_x'] or card['current_x'] or card['eddington_x']:
        card['crossing'] = True; card['crossing_x'] = True
    if card['flux_t'] or card['current_t'] or card['eddington_t'] or card['density_t']:
        card['crossing'] = True; card['crossing_t'] = True

    return card

# ==============================================================================
# Setting
# ==============================================================================

def setting(**kw):
    # Get keyword arguments
    N_hist           = kw.get('N_hist')
    time_boundary    = kw.get('time_boundary')
    rng_seed         = kw.get('rng_seed')
    rng_stride       = kw.get('rng_stride')
    output           = kw.get('output')
    progress_bar     = kw.get('progress_bar')

    # Check if setting card has been initialized
    card = mcdc.input_card.setting

    # Number of histories
    if N_hist is not None:
        card['N_hist'] = int(N_hist)

    # Time boundary
    if time_boundary is not None:
        card['time_boundary'] = time_boundary

    # RNG seed and stride
    if rng_seed is not None:
        card['rng_seed'] = rng_seed
    if rng_stride is not None:
        card['rng_stride'] = rng_stride

    # Output .h5 file name
    if output is not None:
        card['output'] = output

    # Progress bar
    if progress_bar is not None:
        card['progress_bar'] = progress_bar

def eigenmode(N_iter=1, k_init=1.0, alpha_mode=False, alpha_init=0.0,
              gyration_radius='all'):
    # Update setting card
    card                    = mcdc.input_card.setting
    card['N_iter']          = N_iter
    card['mode_eigenvalue'] = True
    card['mode_alpha']      = alpha_mode
    card['k_init']          = k_init
    card['alpha_init']      = alpha_init
    # Gyration radius setup
    if gyration_radius == 'all':
        card['gyration_all'] = True
    elif gyration_radius == 'infinite-z':
        card['gyration_infinite_z'] = True
    elif gyration_radius == 'only-x':
        card['gyration_only_x'] = True
    else:
        print_error("Unknown gyration radius type")

    # Update tally card
    card                = mcdc.input_card.tally
    card['tracklength'] = True

# ==============================================================================
# Technique
# ==============================================================================

def implicit_capture():
    card = mcdc.input_card.technique
    card['implicit_capture'] = True
    card['weighted_emission'] = False

def weighted_emission(flag):
    card = mcdc.input_card.technique
    card['weighted_emission'] = flag

def population_control():
    card = mcdc.input_card.technique
    card['population_control'] = True

def branchless_collision():
    card = mcdc.input_card.technique
    card['branchless_collision'] = True

def weight_window(x=None, y=None, z=None, t=None, window=None):
    card = mcdc.input_card.technique
    card['weight_window'] = True

    # Set mesh
    if x is not None: card['ww_mesh']['x'] = x
    if y is not None: card['ww_mesh']['y'] = y
    if z is not None: card['ww_mesh']['z'] = z
    if t is not None: card['ww_mesh']['t'] = t

    # Set window
    ax_expand = []
    if t is None:
        ax_expand.append(0)
    if x is None:
        ax_expand.append(1)
    if y is None:
        ax_expand.append(2)
    if z is None:
        ax_expand.append(3)
    window /= np.max(window)
    for ax in ax_expand:
        window = np.expand_dims(window, axis=ax)
    card['ww'] = window

    return card

# ==============================================================================
# Util
# ==============================================================================

def print_card(card):
    if isinstance(card, SurfaceHandle):
        card = card.card
    for key in card:
        if key == 'tag':
            print(card[key]+' card')
        else:
            print('  '+key+' : '+str(card[key]))

def universal_speed(speed):
    for C in mcdc.input_card.cells:
        material = mcdc.input_card.materials[C['material_ID']]
        material['speed'] = speed

def universal_decay(decay):
    for C in mcdc.input_card.cells:
        material = mcdc.input_card.materials[C['material_ID']]
        material['decay'] = decay
