# TODO: anisotropic scattering

from numba              import int64, float64, config
from numba.experimental import jitclass

@jitclass([('ID', int64), ('G', int64), ('J', int64), ('speed', float64[:]),
           ('decay', float64[:]), ('total', float64[:]),
           ('capture', float64[:]), ('scatter', float64[:]),
           ('fission', float64[:]), ('nu_s', float64[:]), ('nu_f', float64[:]),
           ('nu_p', float64[:]), ('nu_d', float64[:,:]), ('nu_d', float64[:,:]),
           ('chi_s', float64[:,:]), ('chi_p', float64[:,:]),
           ('chi_d', float64[:,:])])
class Material:
    """
    Attributes
    ----------
    ID : int
        Material ID

    G : int
        # of energy groups
    J : int
        # of delayed neutron precursor groups

    speed : numpy.ndarray (1D)
        Energy group speed
    decay : numpy.ndarray (1D)
        Precursor group decay constant [/s]

    total : numpy.ndarray (1D)
        Total cross-section [/cm]
    capture : numpy.ndarray (1D)
        Capture cross-section [/cm]
    scatter : numpy.ndarray (1D)
        Scattering cross-section [/cm]
    fission : numpy.ndarray (1D)
        Fission cross-section [/cm]

    nu_s : numpy.ndarray (1D)
        Scattering multiplication
    nu_f : numpy.ndarray (1D)
        Total fission neutron yield
    nu_p : numpy.ndarray (1D)
        Prompt fission neutron yield
    nu_d : numpy.ndarray (2D)
        Delayed neutron precursor yield [in][dgroup]

    chi_s : numpy.ndarray (2D)
        Scattering spectrum [in][out]
    chi_p : numpy.ndarray (2D)
        Prompt fission spectrum [in][out]
    chi_d : numpy.ndarray (2D)
        Delayed neutron spectrum [dgroup][out]
    """

    def __init__(self, ID, G, J, speed, decay, total, capture, scatter, fission, 
                 nu_s, nu_f, nu_p, nu_d, chi_s, chi_p, chi_d):
        self.ID      = ID
        self.G       = G
        self.J       = J
        self.speed   = speed
        self.decay   = decay
        self.total   = total
        self.capture = capture
        self.scatter = scatter
        self.fission = fission
        self.nu_s    = nu_s
        self.nu_f    = nu_f
        self.nu_p    = nu_p
        self.nu_d    = nu_d
        self.chi_s   = chi_s
        self.chi_p   = chi_p
        self.chi_d   = chi_d

if not config.DISABLE_JIT:
    type_material = Material.class_type.instance_type
else:
    type_material = None
