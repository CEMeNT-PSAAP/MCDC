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
EVENT_LATTICE        = 10
EVENT_SURFACE_N_MESH = 24
EVENT_LATTICE_N_MESH = 104

# Mesh crossing flags
MESH_X = 0
MESH_Y = 1
MESH_Z = 2
MESH_T = 3

# Misc.
INF   = 1E10
PI    = math.acos(-1.0)
SHIFT = 1E-10 # For particle geometry (surface and mesh) shifts
