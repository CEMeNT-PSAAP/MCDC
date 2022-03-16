import numba as nb

from mcdc.class_.popctrl import *
from mcdc.class_.tally   import Tally
from mcdc.constant       import INF

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

def set_problem(cells, source, N_hist):
    mcdc.cells           = cells
    mcdc.source          = source
    mcdc.settings.N_hist = int(N_hist)

def set_output(output):
    mcdc.settings.output = output

def set_rng(seed=None, stride=None):
    if seed is not None:
        mcdc.settings.seed = seed
    if stride is not None:
        mcdc.settings.stride = stride
    # Random number generator
    mcdc.rng = RandomLCG(mcdc.settings.seed, mcdc.settings.stride)
    
def set_universal_speed(speed):
    for C in mcdc.cells:
        C.material.speed = speed

def set_universal_decay(decay):
    for C in mcdc.cells:
        C.material.decay = decay

def set_tally(scores, x=None, y=None, z=None, t=None):
    mcdc.tally = Tally('tally', scores, x, y, z, t)
    
def set_implicit_capture(flag=True):
    mcdc.settings.implicit_capture = flag

def set_kmode(N_iter, k_init=1.0, alpha_mode=False, alpha_init=0.0):
    mcdc.settings.mode_eigenvalue = True
    mcdc.settings.mode_alpha      = alpha_mode
    mcdc.settings.N_iter          = N_iter
    mcdc.global_tally.k_eff       = k_init
    mcdc.global_tally.alpha_eff   = alpha_init

    # Allocate iterate solution
    mcdc.global_tally.allocate(N_iter)

def set_weight_window(x=None, y=None, z=None, t=None, window=None):
    mcdc.settings.weight_window = WeightWindow(x,y,z,t,window)

def set_population_control(pct='None', census_time=[INF]):
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

    # Set census time
    mcdc.settings.census_time = census_time
