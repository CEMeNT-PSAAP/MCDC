import itertools
import numpy     as np

from mcdc.class_.cell     import Cell
from mcdc.class_.material import Material
from mcdc.class_.point    import Point
from mcdc.class_.source   import *
from mcdc.class_.surface  import Surface, SurfaceHandle
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
    provided through `mcdc.set_universal_speed` and 
    `mcdc.set_universal_decay`.
    """

    # ID
    global material_ID
    material_ID = next(material_count) + 1

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
    
    mcdc.materials.append(
        Material(material_ID, _G, _J, _speed, _decay, _total, _capture, 
                _scatter, _fission, _nu_s, _nu_f, _nu_p, _nu_d,
                _chi_s, _chi_p, _chi_d))
    return mcdc.materials[-1]

#==============================================================================
# Surface
#==============================================================================

surface_ID    = -1
surface_count = itertools.count(0)

def surface(type_, **kw):
    # ID
    next_surface_ID()

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
    # Ax + By + Cz + D = 0
    A = 0.0; B = 0.0; C = 0.0; D = 0.0
    if type_ == 'plane-x':
        A = 1.0
        D = -kw.get('x')
    elif type_ == 'plane-y':
        B = 1.0
        D = -kw.get('y')
    elif type_ == 'plane-z':
        C = 1.0
        D = -kw.get['z']
    elif type_ == 'plane':
        A = -kw.get('A')
        B = -kw.get('B')
        C = -kw.get('C')
        D = -kw.get('D')
    else:
        print_error("Unsupported surface type: "+type_)

    # Create object
    mcdc.surfaces.append(Surface(surface_ID, vacuum, reflective, A, B, C, D))
    return SurfaceHandle(mcdc.surfaces[-1])

def next_surface_ID():
    global surface_ID
    surface_ID = next(surface_count) + 1

def get_bc(bc):
    vacuum     = False
    reflective = False
    if bc == '':
        pass
    elif bc == 'vacuum':
        vacuum = True
    elif bc == 'reflective':
        reflective = True
    else:
        print_error("Undefined surface boundary condition:"+bc)
    return vacuum, reflective

#==============================================================================
# Cell
#==============================================================================

cell_ID    = -1
cell_count = itertools.count(0)

def cell(surfaces_senses, material):
    # ID
    global cell_ID
    cell_ID = next(cell_count) + 1

    # Surfaces and senses
    surfaces = []
    senses   = []
    for s in surfaces_senses:
        surfaces.append(s[0])
        senses.append(s[1])

    mcdc.cells.append(Cell(cell_ID, surfaces, senses, material))
    return mcdc.cells[-1]

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

    # Set direction
    if isotropic is None:
        isotropic = False
        if direction is None:
            isotropic = True
        else:
            direction = Point(direction[0], direction[1], direction[2])
            direction.normalize()

    # Set energy
    if energy is None:
        energy = np.array([1.0])
    else:
        energy /= sum(energy)

    # Set time
    if time is None:
        time = np.array([0.0, 0.0])

    # Set source
    if point is not None:
        x = point[0]
        y = point[1]
        z = point[2]
        position = Point(x,y,z)
        if isotropic:
            return SourcePointIsotropic(position, energy, time)
        else:
            return SourcePointMono(position, direction, energy, time)
    else:
        if x is None:
            x = [0.0, 0.0]
        if y is None:
            y = [0.0, 0.0]
        if z is None:
            z = [0.0, 0.0]
        box = np.array([x,y,z])
        if isotropic:
            return SourceBoxIsotropic(box, energy, time)
        else:
            return SourceBoxMono(box, direction, energy, time)
