from   numba              import int64, float64
from   numba.experimental import jitclass
import numpy              as     np

class Particle:
    def __init__(self, position, direction, group, time, weight):
        # Particle phase space
        self.position  = position  # cm
        self.direction = direction
        self.group     = group
        self.time      = time # s
        self.weight    = weight

        # Flags
        self.alive = True

        # Tags
        self.cell = None
        self.idx_census_time = None

        # Misc
        self.speed = 1.0 # Updated in particle loop

    def copy(self):
        P = Particle(self.position.copy(), self.direction.copy(), self.group, 
                     self.time, self.weight)

        # Copy tags
        P.cell            = self.cell
        P.idx_census_time = self.idx_census_time
        return P
