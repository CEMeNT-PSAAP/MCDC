import math
import numpy as np

# Events
EVENT_COLLISION           = 1
EVENT_SURFACE             = 2
EVENT_CENSUS              = 3
EVENT_MESH                = 4
EVENT_SURFACE_N_MESH      = 40
EVENT_SURFACE_MOVE        = 41
EVENT_SURFACE_MOVE_N_MESH = 42
EVENT_SCATTERING          = 5
EVENT_FISSION             = 6
EVENT_CAPTURE             = 7
EVENT_TIME_BOUNDARY       = 8
EVENT_LATTICE             = 9
EVENT_LATTICE_N_MESH      = 90

# Mesh crossing flags
MESH_X = 0
MESH_Y = 1
MESH_Z = 2
MESH_T = 3

# Gyration raius type
GR_ALL        = 0
GR_INFINITE_X = 1
GR_INFINITE_Y = 2
GR_INFINITE_Z = 3
GR_ONLY_X     = 4
GR_ONLY_Y     = 5
GR_ONLY_Z     = 6

# Misc.
INF   = 1E10
PI    = math.acos(-1.0)
SHIFT = 1E-10 # To ensure lattice, surface, and mesh crossings
PREC  = 1.0+1E-5 # Precision factor to determine if a distance is smaller than
                 # another (for lattice, surface, and mesh)
BANKMAX = 100 # Default maximum active bank
