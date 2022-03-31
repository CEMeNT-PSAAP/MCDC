import itertools
from   numba.typed import List
import numpy       as np

from mcdc.class_.cell     import Cell
from mcdc.class_.material import Material
from mcdc.class_.mesh     import Mesh
from mcdc.class_.point    import Point
from mcdc.class_.popctrl  import *
from mcdc.class_.source   import Source
from mcdc.class_.surface  import Surface, SurfaceHandle
from mcdc.class_.tally    import Tally
from mcdc.constant        import INF
       
# Get mcdc global variables/objects
import mcdc.global_ as mcdc

#==============================================================================
# Material
#==============================================================================

material_ID    = -1
material_count = itertools.count(0)

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

    # ID
    global material_ID
    material_ID = next(material_count)

    # Energy group size
    if capture is not None:
        _G = len(capture)
    elif scatter is not None:
        _G = len(scatter)
    elif fission is not None:
        _G = len(fission)
    else:
        print_error("Need to supply capture, scatter, or fission to "\
                    + "mcdc.material")

    # Delayed group size
    if nu_d is not None:
        _J = len(nu_d)
    else:
        _J = 1
    
    # Cross-sections
    if capture is not None:
        _capture = capture
    else:
        _capture = np.zeros(_G)
    if scatter is not None:
        _scatter = np.sum(scatter,0)
    else:
        _scatter = np.zeros(_G)
    if fission is not None:
        if fission.ndim == 1:
            _fission = fission
        else:
            _fission = np.sum(fission,0)
    else:
        _fission = np.zeros(_G)
    
    # Scattering multiplication
    if nu_s is not None:
        _nu_s = nu_s
    else:
        _nu_s = np.ones(_G)

    # Fission productions
    if fission is not None:
        if nu_p is None and nu_d is None:
            print_error("Need to supply nu_p or nu_d for fissionable "\
                        + "mcdc.material")
    if nu_p is not None:
        _nu_p = nu_p
    else:
        _nu_p = np.zeros(_G)
    if nu_d is not None:
        _nu_d  = np.copy(nu_d)
        _nu_d  = np.swapaxes(_nu_d,0,1) # [dg,gin] -> [gin,dg]
    else:
        _nu_d = np.zeros([_G,1])
    _nu_f  = np.zeros(_G)
    _nu_f += _nu_p
    for j in range(_J):
        _nu_f += _nu_d[:,j]

    # Scattering spectrum
    if scatter is not None:
        _chi_s = np.copy(scatter)
        _chi_s = np.swapaxes(_chi_s,0,1) # [gout,gin] -> [gin,gout]
        for g in range(_G): 
            if _scatter[g] > 0.0:
                _chi_s[g,:] /= _scatter[g]
    else:
        _chi_s = np.zeros([_G,_G])

    # Fission spectrums
    if nu_p is not None:
        if _G == 1:
            _chi_p = np.array([[1.0]])
        elif chi_p is None:
            print_error("Need to supply chi_p if nu_p is provided")
        else:
            _chi_p = np.copy(chi_p)
            _chi_p = np.swapaxes(_chi_p,0,1) # [gout,gin] -> [gin,gout]
            # Normalize
            for g in range(_G):
                if np.sum(_chi_p[g,:]) > 0.0:
                    _chi_p[g,:] /= np.sum(_chi_p[g,:])
    else:
        _chi_p = np.zeros([_G,_G])
    if nu_d is not None:
        if chi_d is None:
            print_error("Need to supply chi_d if nu_d is provided")
        _chi_d = np.copy(chi_d)
        _chi_d = np.swapaxes(_chi_d,0,1) # [gout,dg] -> [dg,gout]
        # Normalize
        for dg in range(_J):
            if np.sum(_chi_d[dg,:]) > 0.0:
                _chi_d[dg,:] /= np.sum(_chi_d[dg,:])
    else:
        _chi_d = np.zeros([_G,_J])

    # Get speed
    if speed is not None:
        _speed = speed
    else:
        _speed = np.ones(_G)

    # Get decay constant
    if decay is not None:
        _decay = decay
    else:
        _decay = np.ones(_J)*np.inf

    # Total
    _total = _capture + _scatter + _fission
   
    # Create object
    M = Material(material_ID, _G, _J, _speed, _decay, _total, _capture, 
                 _scatter, _fission, _nu_s, _nu_f, _nu_p, _nu_d, _chi_s, 
                 _chi_p, _chi_d)
    mcdc.global_.materials.append(M)
    return M

#==============================================================================
# Surface
#==============================================================================

surface_ID    = -1
surface_count = itertools.count(0)

def surface(type_, **kw):
    # ID
    global surface_ID
    surface_ID = next(surface_count)

    # Boundary condition
    vacuum     = False
    reflective = False
    bc = kw.get('bc')
    if bc is not None:
        if bc == 'vacuum':
            vacuum = True
        elif bc == 'reflective':
            reflective = True
        else:
            print_error("Unsupported surface boundary condition: "+bc)
    # Type
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0
    F = 0.0; G = 0.0; H = 0.0; I = 0.0; J = 0.0
    if type_ == 'plane-x':
        G = 1.0
        J = -kw.get('x')
    elif type_ == 'plane-y':
        H = 1.0
        J = -kw.get('y')
    elif type_ == 'plane-z':
        I = 1.0
        J = -kw.get('z')
    elif type_ == 'plane':
        G = kw.get('A')
        H = kw.get('B')
        I = kw.get('C')
        J = kw.get('D')
    elif type_ == 'cylinder-x':
        y, z = kw.get('center')[:]
        r    = kw.get('radius')
        B = 1.0
        C = 1.0
        H = -2.0*y
        I = -2.0*z
        J = y**2 + z**2 - r**2
    elif type_ == 'cylinder-y':
        x, z = kw.get('center')[:]
        r    = kw.get('radius')
        A = 1.0
        C = 1.0
        G = -2.0*x
        I = -2.0*z
        J = x**2 + z**2 - r**2
    elif type_ == 'cylinder-z':
        x, y = kw.get('center')[:]
        r    = kw.get('radius')
        A = 1.0
        B = 1.0
        G = -2.0*x
        H = -2.0*y
        J = x**2 + y**2 - r**2
    elif type_ == 'sphere':
        x, y, z = kw.get('center')[:]
        r       = kw.get('radius')
        A = 1.0
        B = 1.0
        C = 1.0
        G = -2.0*x
        H = -2.0*y
        I = -2.0*z
        J = x**2 + y**2 + z**2 - r**2
    elif type_ == 'quadric':
        A = kw.get('A')
        B = kw.get('B')
        C = kw.get('C')
        D = kw.get('D')
        E = kw.get('E')
        F = kw.get('F')
        G = kw.get('G')
        H = kw.get('H')
        I = kw.get('I')
        J = kw.get('J')
    else:
        print_error("Unsupported surface type: "+type_)

    # Create object
    S = Surface(surface_ID, vacuum, reflective, A, B, C, D, E, F, G, H, I, J)
    mcdc.global_.surfaces.append(S)
    return SurfaceHandle(S)

#==============================================================================
# Cell
#==============================================================================

cell_ID    = -1
cell_count = itertools.count(0)

def cell(surfaces_senses, material):
    # ID
    global cell_ID
    cell_ID = next(cell_count)

    # Surfaces and senses
    surfaces = List()
    senses   = []
    for s in surfaces_senses:
        surfaces.append(s[0])
        senses.append(s[1])

    # Set appropriate type
    senses = np.array(senses)

    # Create object
    C = Cell(cell_ID, surfaces, senses, material)
    mcdc.global_.cells.append(C)

#==============================================================================
# Source
#==============================================================================

def source(**kw):
    point     = kw.get('point') # Point source
    x         = kw.get('x')     # Box source
    y         = kw.get('y')
    z         = kw.get('z')
    isotropic = kw.get('isotropic')
    direction = kw.get('direction') # Mono-directional
    energy    = kw.get('energy')
    time      = kw.get('time')

    # Get object
    S = mcdc.global_.source

    # Set position
    if point is not None:
        x = point[0]
        y = point[1]
        z = point[2]
        S.position = Point(x,y,z)
    else:
        S.flag_box = True
        box = []
        if x is None:
            x = [0.0, 0.0]
        box.extend(x)
        if y is None:
            y = [0.0, 0.0]
        box.extend(y)
        if z is None:
            z = [0.0, 0.0]
        box.extend(z)
        S.box = np.array(box, dtype=np.float64)

    # Set direction
    if direction is not None:
        S.flag_isotropic = False
        direction = Point(direction[0], direction[1], direction[2])
        direction.normalize()
        S.direction = direction

    # Set energy
    if energy is not None:
        energy = np.array(energy)
        energy /= np.sum(energy)
        S.energy = energy

    # Set time
    if time is not None:
        S.time = time

#==============================================================================
# Tally
#==============================================================================

def tally(scores, x=None, y=None, z=None, t=None):
    # Get object
    T = mcdc.global_.tally

    # Check scores
    for s in scores:
        if s == 'flux':
            T.flux = True
        elif s == 'current':
            T.current = True
        elif s == 'eddington':
            T.eddington = True
        else:
            print_error("Unknown tally score %s"%s)
    mesh = set_mesh(x,y,z,t)
    T.mesh = mesh

def set_mesh(x, y, z, t):
    if x is None:
        x = np.array([-INF,INF])
    else:
        x = x
    if y is None:
        y = np.array([-INF,INF])
    else:
        y = y
    if z is None:
        z = np.array([-INF,INF])
    else:
        z = z
    if t is None:
        t = np.array([-INF,INF])
    else:
        t = t
    return Mesh(x,y,z,t)

#==============================================================================
# Setting
#==============================================================================

def setting(**kw):
    N_hist = kw.get('N_hist')
    output = kw.get('output')
    seed  = kw.get('seed')
    stride = kw.get('stride')
    implicit_capture = kw.get('implicit_capture')
    time_boundary = kw.get('time_boundary')
    progress_bar  = kw.get('progress_bar')

    # Get object
    S = mcdc.global_.setting

    # Number of histories
    if N_hist is not None:
        S.N_hist = int(N_hist)

    # Output .h5 file name
    if output is not None:
        S.output = output

    # RNG seed and stride
    if seed is not None:
        mcdc.global_.rng.set_seed(seed)
    if stride is not None:
        mcdc.global_.rng.set_stride(stride)

    # Variance reduction technique
    if implicit_capture is not None:
        S.implicit_capture = True

    # Time boundary
    if time_boundary is not None:
        S.time_boundary = time_boundary

    # Progress bar
    if progress_bar is not None:
        S.progress_bar = progress_bar
    
def universal_speed(speed):
    for C in mcdc.global_.cells:
        C.material.speed = speed

def universal_decay(decay):
    for C in mcdc.global_.cells:
        C.material.decay = decay

def eigenmode(N_iter=1, k_init=1.0, alpha_mode=False, alpha_init=0.0):
    # Get object
    S = mcdc.global_.setting
    GT = mcdc.global_.tally_global

    S.N_iter          = N_iter
    S.mode_eigenvalue = True
    S.mode_alpha      = alpha_mode

    GT.k_eff     = k_init
    GT.alpha_eff = alpha_init

    # Allocate global tally
    GT.allocate(N_iter)

def weight_window(x=None, y=None, z=None, t=None, window=None):
    mcdc.global_.setting.weight_window = True

    # Set mesh
    mesh = set_mesh(x,y,z,t)

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
    if window is None:
        print_error('Window is missing for weight window')
    window /= np.max(window)
    for ax in ax_expand:
        window = np.expand_dims(window, axis=ax)

    mcdc.global_.weight_window = WeightWindow(mesh,window)

def population_control(pct):
    # Set technique
    if pct in ['SS', 'simple-sampling']:
        mcdc.population_control = PCT_SS()
    elif pct in ['SR', 'splitting-roulette']:
        mcdc.population_control = PCT_SR()
    elif pct in ['CO', 'combing']:
        mcdc.population_control = PCT_CO()
    elif pct in ['COX', 'combing-modified']:
        mcdc.population_control = PCT_COX()
    elif pct in ['DD', 'duplicate-discard']:
        mcdc.population_control = PCT_DD()
    elif pct in ['None']:
        mcdc.population_control = PCT_None()
