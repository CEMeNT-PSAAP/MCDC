import numpy as np

from   mcdc.class_.popctrl import PCT_None
from   mcdc.constant       import INF, LCG_SEED, LCG_STRIDE
import mcdc.mpi            as     mpi
from   mcdc.class_.random  import RandomLCG

#==============================================================================
# Problem definition
#==============================================================================
# TODO: use fixed memory allocations with helper indices

cells   = []
sources = []
tally   = None


#==============================================================================
# Particle banks
#==============================================================================

bank_stored  = []   # for the next source loop
bank_source  = []   # for current source loop
bank_history = []   # for current history loop
bank_fission = None # will point to history/stored bank


#==============================================================================
# Settings
#==============================================================================

class Settings:
    def __init__(self):
        # Basic settings
        self.N_hist = 0
        self.N_iter = 1 # For eigenvalue mode
        self.output = "output" # .h5 output file name

        # Random number generator
        self.seed   = LCG_SEED
        self.stride = LCG_STRIDE

        # Mode flags
        self.mode_eigenvalue = False
        self.mode_alpha      = False

        # Variance reduction technique
        self.implicit_capture   = False
        self.weight_window      = False

        # Time census
        self.census_time = [INF]

        # Time boundary
        self.time_boundary = INF # TODO

        # Misc.
        self.parallel_hdf5 = False # TODO

settings = Settings()


#==============================================================================
# Global tally
#==============================================================================
# TODO: alpha eigenvalue with delayed neutrons
# TODO: shannon entropy

class GlobalTally:
    def __init__(self):
        # Effective eigenvalue
        self.k_eff     = 1.0
        if settings.mode_alpha:
            self.alpha_eff = 0.0

        # Accumulator
        self.nuSigmaF_sum = 0.0
        self.ispeed_sum = 0.0

        # MPI buffer
        self.nuSigmaF_buff = np.array([0.0])
        self.ispeed_buff = np.array([0.0])

    def allocate(self, N_iter):
        # Eigenvalue solution iterate
        if mpi.master:
            self.k_mean = np.zeros(N_iter)
            if settings.mode_alpha:
                self.alpha_mean = np.zeros(N_iter)

global_tally = GlobalTally()


#==============================================================================
# Variance reduction techniques
#==============================================================================

population_control = PCT_None()
weight_window      = None

#==============================================================================
# Runtimer
#==============================================================================

class Runtimer:
    def __init__(self):
        self.total = 0.0
        self.click = 0.0

    def start(self):
        self.click = mpi.Wtime()

    def stop(self):
        self.total += mpi.Wtime() - self.click

runtime_total = Runtimer()


#==============================================================================
# Random number generator
#==============================================================================

rng = RandomLCG()
