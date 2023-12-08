import math
import numpy as np
import numba as nb

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

# Gyration raius type
GYRATION_RADIUS_ALL = 0
GYRATION_RADIUS_INFINITE_X = 1
GYRATION_RADIUS_INFINITE_Y = 2
GYRATION_RADIUS_INFINITE_Z = 3
GYRATION_RADIUS_ONLY_X = 4
GYRATION_RADIUS_ONLY_Y = 5
GYRATION_RADIUS_ONLY_Z = 6

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

# RNG LCG parameters
RNG_G = nb.uint64(2806196910506780709)
RNG_C = nb.uint64(1)
RNG_MOD_MASK = nb.uint64(0x7FFFFFFFFFFFFFFF)
RNG_MOD = nb.uint64(0x8000000000000000)

# RNG splitter seeds
SEED_SPLIT_CENSUS = nb.uint64(0x43454D654E54)
SEED_SPLIT_SOURCE = nb.uint64(0x43616D696C6C65)
SEED_SPLIT_SOURCE_PRECURSOR = nb.uint64(0x546F6464)
SEED_SPLIT_BANK = nb.uint64(0x5279616E)
SEED_SPLIT_PARTICLE = nb.uint64(0)
SEED_SPLIT_UQ = nb.uint64(0x5368656261)
