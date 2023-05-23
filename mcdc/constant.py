import math
import numpy as np

# Events
# The << operator represents a bitshift. Each event is assigned 1 << X, which is equal to 2 to the power of X.
EVENT_COLLISION = 1 << 0
EVENT_SURFACE = 1 << 1
EVENT_CENSUS = 1 << 2
EVENT_MESH = 1 << 3
EVENT_SCATTERING = 1 << 4
EVENT_FISSION = 1 << 5
EVENT_CAPTURE = 1 << 6
EVENT_TIME_BOUNDARY = 1 << 7
EVENT_LATTICE = 1 << 8
EVENT_SURFACE_MOVE = 1 << 9

# Mesh crossing flags
MESH_X = 0
MESH_Y = 1
MESH_Z = 2
MESH_T = 3

# Gyration raius type
GR_ALL = 0
GR_INFINITE_X = 1
GR_INFINITE_Y = 2
GR_INFINITE_Z = 3
GR_ONLY_X = 4
GR_ONLY_Y = 5
GR_ONLY_Z = 6

# Population control
PCT_NONE = 0
PCT_COMBING = 1
PCT_COMBING_WEIGHT = 10

# Misc.
INF = 1e10
PI = math.acos(-1.0)
SHIFT = 1e-10  # To ensure lattice, surface, and mesh crossings
PREC = 1.0 + 1e-5  # Precision factor to determine if a distance is smaller
BANKMAX = 100  # Default maximum active bank
