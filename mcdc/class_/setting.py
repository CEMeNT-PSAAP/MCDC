from numba              import int64, float64, boolean, config
from numba.types        import string
from numba.experimental import jitclass

from mcdc.constant import INF

@jitclass([('N_hist', int64), ('N_iter', int64), 
           ('mode_eigenvalue', boolean), ('mode_alpha', boolean), 
           ('implicit_capture', boolean), ('weight_window', boolean),
           ('time_boundary', float64), ('parallel_hdf5', boolean),
           ('output_name', string), ('progress_bar', boolean)])
class Setting:
    def __init__(self):
        # Basic settings
        self.N_hist = 0
        self.N_iter = 1 # For eigenvalue mode

        # Mode flags
        self.mode_eigenvalue = False
        self.mode_alpha      = False

        # Variance reduction technique
        self.implicit_capture = False
        self.weight_window    = False

        # Time boundary
        self.time_boundary = INF

        # Misc.
        self.parallel_hdf5 = False # TODO
        self.output_name   = 'output' # .h5 output file name
        self.progress_bar  = True

if not config.DISABLE_JIT:
    type_setting = Setting.class_type.instance_type
else:
    type_setting = None
