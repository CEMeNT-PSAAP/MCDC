import h5py
import numpy as np

import mcdc.type_ as type_

from mcdc.card     import SurfaceHandle
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
        card['nu_s'][:] = nu_s[:]

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
    card['J']          = np.array([[0.0, 0.0]])
    card['t']          = np.array([-SHIFT, INF])
    card['N_slice']    = 1
    card['linear']     = False
    card['nx']         = 0.0
    card['ny']         = 0.0
    card['nz']         = 0.0

    # Boundary condition
    bc = kw.get('bc')
    if bc is not None:
        bc = bc.lower()
        if bc == 'vacuum':
            card['vacuum'] = True
        elif bc == 'reflective':
            card['reflective'] = True
        else:
            print_error("Unsupported surface boundary condition: "+bc+ '; Supported options are "vacuum" or "reflective"')
    # Surface type
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
    #   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0
    type_ = type_.replace('_','-').replace(' ','-').lower()
    if type_ == 'plane-x':
        card['G']      = 1.0
        card['linear'] = True
        if type(kw.get('x')) in [type([]), type(np.array([]))]:
            set_J(kw.get('x'), kw.get('t'), card)
        else:
            card['J'][0,0] = -kw.get('x')
    elif type_ == 'plane-y':
        card['H']      = 1.0
        card['linear'] = True
        if type(kw.get('y')) in [type([]), type(np.array([]))]:
            set_J(kw.get('y'), kw.get('t'), card)
        else:
            card['J'][0,0] = -kw.get('y')
    elif type_ == 'plane-z':
        card['I']      = 1.0
        card['linear'] = True
        if type(kw.get('z')) in [type([]), type(np.array([]))]:
            set_J(kw.get('z'), kw.get('t'), card)
        else:
            card['J'][0,0] = -kw.get('z')
    elif type_ == 'plane':
        card['G'] = kw.get('A')
        card['H'] = kw.get('B')
        card['I'] = kw.get('C')
        card['J'][0,0] = kw.get('D')
        card['linear'] = True
    elif type_ == 'cylinder-x':
        y, z = kw.get('center')[:]
        r    = kw.get('radius')
        card['B'] = 1.0
        card['C'] = 1.0
        card['H'] = -2.0*y
        card['I'] = -2.0*z
        card['J'][0,0] = y**2 + z**2 - r**2
    elif type_ == 'cylinder-y':
        x, z = kw.get('center')[:]
        r    = kw.get('radius')
        card['A'] = 1.0
        card['C'] = 1.0
        card['G'] = -2.0*x
        card['I'] = -2.0*z
        card['J'][0,0] = x**2 + z**2 - r**2
    elif type_ == 'cylinder-z':
        x, y = kw.get('center')[:]
        r    = kw.get('radius')
        card['A'] = 1.0
        card['B'] = 1.0
        card['G'] = -2.0*x
        card['H'] = -2.0*y
        card['J'][0,0] = x**2 + y**2 - r**2
    elif type_ == 'sphere':
        x, y, z = kw.get('center')[:]
        r       = kw.get('radius')
        card['A'] = 1.0
        card['B'] = 1.0
        card['C'] = 1.0
        card['G'] = -2.0*x
        card['H'] = -2.0*y
        card['I'] = -2.0*z
        card['J'][0,0] = x**2 + y**2 + z**2 - r**2
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
        card['J'][0,0] = kw.get('J')
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

def set_J(x, t, card):
    # Edit and add the edges
    t[0] = -SHIFT
    t    = np.append(t, INF)
    x    = np.append(x, x[-1])

    # Reset the constants
    card['J'] = np.zeros([0,2])
    card['t'] = np.array([-SHIFT])

    # Iterate over inputs
    idx = 0
    for i in range(len(t)-1):
        # Skip if step
        if t[i] == t[i+1]:
            continue

        # Calculate constants
        J0 = x[i]
        J1 = (x[i+1]-x[i])/(t[i+1]-t[i])

        # Append to card
        card['J'] = np.append(card['J'], [[J0, J1]], axis=0)
        card['t'] = np.append(card['t'], t[i+1])

    card['J'] *= -1
    card['N_slice'] = len(card['J'])

# =============================================================================
# Cell
# =============================================================================

def cell(surfaces_flags, fill, lattice_center=None):
    N_surface = len(surfaces_flags)

    # Set default card values (c.f. type_.py)
    card                   = {}
    card['tag']            = 'Cell'
    card['ID']             = len(mcdc.input_card.cells)
    card['N_surface']      = N_surface
    card['surface_IDs']    = np.zeros(N_surface, dtype=int)
    card['positive_flags'] = np.zeros(N_surface, dtype=bool)
    card['material_ID']    = 0
    card['lattice']        = False
    card['lattice_ID']     = 0
    card['lattice_center'] = np.array([0.0, 0.0, 0.0])

    # Surfaces and flags
    for i in range(N_surface):
        card['surface_IDs'][i]    = surfaces_flags[i][0]['ID']
        card['positive_flags'][i] = surfaces_flags[i][1]

    # Lattice cell?
    if fill['tag'] == 'Lattice':
        card['lattice']    = True
        card['lattice_ID'] = fill['ID']
        if lattice_center is not None:
            card['lattice_center'] = np.array(lattice_center)

    # Material cell
    else:
        card['material_ID'] = fill['ID']

    # Push card
    mcdc.input_card.cells.append(card)
    return card

# =============================================================================
# Universe
# =============================================================================

def universe(cells, root=False):
    N_cell = len(cells)

    # Set default card values (c.f. type_.py)
    if not root:
        card        = {}
        card['tag'] = 'Universe'
    else:
        card = mcdc.input_card.universes[0]
    card['ID']       = len(mcdc.input_card.universes)
    card['N_cell']   = N_cell
    card['cell_IDs'] = np.zeros(N_cell, dtype=int)

    # Cells
    for i in range(N_cell):
        card['cell_IDs'][i] = cells[i]['ID']

    # Push card
    if not root:
        mcdc.input_card.universes.append(card)

    return card

#==============================================================================
# Lattice
#==============================================================================

def lattice(x=None, y=None, z=None, universes=None):
    # Set default card values (c.f. type_.py)
    card                 = {}
    card['tag']          = 'Lattice'
    card['ID']           = len(mcdc.input_card.lattices)
    card['mesh']         = {'x0' : -INF, 'dx' : 2*INF, 'Nx' : 1,
                            'y0' : -INF, 'dy' : 2*INF, 'Ny' : 1,
                            'z0' : -INF, 'dz' : 2*INF, 'Nz' : 1}
    card['universe_IDs'] = np.array([[[[0]]]])

    # Set mesh
    if x is not None:
        card['mesh']['x0'] = x[0]
        card['mesh']['dx'] = x[1]
        card['mesh']['Nx'] = x[2]
    if y is not None:
        card['mesh']['y0'] = y[0]
        card['mesh']['dy'] = y[1]
        card['mesh']['Ny'] = y[2]
    if z is not None:
        card['mesh']['z0'] = z[0]
        card['mesh']['dz'] = z[1]
        card['mesh']['Nz'] = z[2]

    # Set universe IDs
    universe_IDs = np.array(universes, dtype=np.int64)
    ax_expand = []
    if x is None:
        ax_expand.append(2)
    if y is None:
        ax_expand.append(1)
    if z is None:
        ax_expand.append(0)
    for ax in ax_expand:
        universe_IDs = np.expand_dims(universe_IDs, axis=ax)

    # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
    tmp = np.transpose(universe_IDs)
    tmp = np.flip(tmp, axis=1)
    card['universe_IDs'] = np.flip(tmp, axis=2)

    # Push card
    mcdc.input_card.lattices.append(card)
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
    white     = kw.get('white_direction')
    energy    = kw.get('energy')
    time      = kw.get('time')
    prob      = kw.get('prob')
    # Set default card values (c.f. type_.py)
    card              = {}
    card['tag']       = 'Source'
    card['ID']        = len(mcdc.input_card.sources)
    card['box']       = False
    card['isotropic'] = True
    card['white']     = False
    card['x']         = 0.0
    card['y']         = 0.0
    card['z']         = 0.0
    card['box_x']     = np.array([0.0, 0.0])
    card['box_y']     = np.array([0.0, 0.0])
    card['box_z']     = np.array([0.0, 0.0])
    card['ux']        = 0.0
    card['uy']        = 0.0
    card['uz']        = 0.0
    card['white_x']   = 0.0
    card['white_y']   = 0.0
    card['white_z']   = 0.0
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
    if white is not None:
        card['isotropic'] = False
        card['white']     = True
        ux = white[0]
        uy = white[1]
        uz = white[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2)**0.5
        card['white_x'] = ux/norm
        card['white_y'] = uy/norm
        card['white_z'] = uz/norm
    elif direction is not None:
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

def tally(scores, x=np.array([-INF, INF]), y=np.array([-INF, INF]), 
          z=np.array([-INF, INF]), t=np.array([-INF, INF]), 
          mu=np.array([-1.0, 1.0]), azi=np.array([-PI, PI])):

    # Get tally card
    card = mcdc.input_card.tally

    # Set mesh
    card['mesh']['x']   = x
    card['mesh']['y']   = y
    card['mesh']['z']   = z
    card['mesh']['t']   = t
    card['mesh']['mu']  = mu
    card['mesh']['azi'] = azi

    # Set score flags
    for s in scores:
        found = False
        for score_name in type_.score_list:
            if s.replace('-','_') == score_name:
                card[score_name] = True
                found = True
                break
        if not found:
            print_error("Unknown tally score %s"%s)

    # Set estimator flags
    for score_name in type_.score_tl_list:
        if card[score_name]:
            card['tracklength'] = True
            break
    for score_name in type_.score_x_list:
        if card[score_name]:
            card['crossing'] = True
            card['crossing_x'] = True
            break
    for score_name in type_.score_y_list:
        if card[score_name]:
            card['crossing'] = True
            card['crossing_y'] = True
            break
    for score_name in type_.score_z_list:
        if card[score_name]:
            card['crossing'] = True
            card['crossing_z'] = True
            break
    for score_name in type_.score_t_list:
        if card[score_name]:
            card['crossing'] = True
            card['crossing_t'] = True
            break

    return card

# ==============================================================================
# Setting
# ==============================================================================

def setting(**kw):
    # Get keyword arguments
    N_particle       = kw.get('N_particle')
    time_boundary    = kw.get('time_boundary')
    rng_seed         = kw.get('rng_seed')
    rng_stride       = kw.get('rng_stride')
    output           = kw.get('output')
    progress_bar     = kw.get('progress_bar')
    k_eff            = kw.get('k_eff')
    bank_active_buff = kw.get('active_bank_buff')
    bank_census_buff = kw.get('census_bank_buff')
    source_file      = kw.get('source_file')

    # Check if setting card has been initialized
    card = mcdc.input_card.setting

    # Number of particles
    if N_particle is not None:
        card['N_particle'] = int(N_particle)

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

    # k effective
    if k_eff is not None:
        card['k_init'] = k_eff

    # Maximum active bank size
    if bank_active_buff is not None:
        card['bank_active_buff'] = int(bank_active_buff)

    # Census bank size multiplier
    if bank_census_buff is not None:
        card['bank_census_buff'] = int(bank_census_buff)

def eigenmode(N_inactive=0, N_active=0, k_init=1.0, gyration_radius=None,
              N_cycle_buff=0):
    # Update setting card
    card                    = mcdc.input_card.setting
    card['N_inactive']      = N_inactive
    card['N_active']        = N_active
    card['N_cycle']         = N_inactive + N_active
    card['mode_eigenvalue'] = True
    card['k_init']          = k_init
    card['N_cycle_buff']    = N_cycle_buff

    # Gyration radius setup
    if gyration_radius is not None:
        card['gyration_radius'] = True
        if gyration_radius == 'all':
            card['gyration_radius_type'] = GR_ALL
        elif gyration_radius == 'infinite-x':
            card['gyration_radius_type'] = GR_INFINITE_X
        elif gyration_radius == 'infinite-y':
            card['gyration_radius_type'] = GR_INFINITE_Y
        elif gyration_radius == 'infinite-z':
            card['gyration_radius_type'] = GR_INFINITE_Z
        elif gyration_radius == 'only-x':
            card['gyration_radius_type'] = GR_ONLY_X
        elif gyration_radius == 'only-y':
            card['gyration_radius_type'] = GR_ONLY_Y
        elif gyration_radius == 'only-z':
            card['gyration_radius_type'] = GR_ONLY_Z
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

def population_control(pct='combing'):
    card = mcdc.input_card.technique
    card['population_control'] = True
    card['weighted_emission']  = False
    if pct == 'combing':
        card['pct'] = PCT_COMBING
    elif pct == 'combing-weight':
        card['pct'] = PCT_COMBING_WEIGHT
    else:
        print_error("Unknown PCT type " + pct)

def branchless_collision():
    card = mcdc.input_card.technique
    card['branchless_collision'] = True
    card['weighted_emission'] = False

def census(t, pct='none'):
    card = mcdc.input_card.technique
    card['time_census'] = True
    card['census_time'] = t
    if pct != 'none':
        population_control(pct)

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

def IC_generator(N_neutron=0, N_precursor=0):
    card = mcdc.input_card.technique
    card['IC_generator']   = True
    card['IC_N_neutron']   = int(N_neutron)
    card['IC_N_precursor'] = int(N_precursor)


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
