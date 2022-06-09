import math
import numpy as np

# Events
EVENT_COLLISION      = 1
EVENT_SURFACE        = 2
EVENT_CENSUS         = 3
EVENT_MESH           = 4
EVENT_SCATTERING     = 5
EVENT_FISSION        = 6
EVENT_CAPTURE        = 7
EVENT_TIME_REACTION  = 8
EVENT_TIME_BOUNDARY  = 9
EVENT_SURFACE_N_MESH = 24

# Population Control Techniques
PCT_SS  = 1
PCT_SR  = 2
PCT_CO  = 3
PCT_COX = 4
PCT_DD  = 5

# Misc.
INF       = 1E10
PI        = math.acos(-1.0)
EPSILON   = 1E-15 # To ensure non-zero value
PRECISION = 1E-8  # Used in surface crossing
