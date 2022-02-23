from abc import ABC, abstractmethod

from mcdc.point import Point
from mcdc.print import print_error

import mcdc


# =============================================================================
# Particle
# =============================================================================

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

    def create_copy(self):
        P = Particle(self.position, self.direction, self.group, self.time, 
                     self.weight)

        # Copy tags
        P.cell            = self.cell
        P.idx_census_time = self.idx_census_time
        return P
        
# =============================================================================
# Source
# =============================================================================

class Source(ABC):
    def __init__(self, prob):
        self.prob = prob

    @abstractmethod
    def get_particle(self):
        pass

class SourceSimple(Source):
    def __init__(self, position=None, direction=None, group=None, time=None,
                 prob=1.0):

        Source.__init__(self, prob)
        if position is None:
            self.position = mcdc.DistPoint()
        else:
            self.position = position
        if direction is None:
            self.direction = mcdc.DistPointIsotropic()
        else:
            self.direction = direction
        if group is None:
            self.group = mcdc.DistDelta(0)
        else:
            self.group = group

        if time is None:
            self.time = mcdc.DistDelta(0.0)
        else:
            self.time = time

    def get_particle(self):
        position  = self.position.sample()
        direction = self.direction.sample()
        group     = self.group.sample()
        time      = self.time.sample()
        direction.normalize()
        return Particle(position, direction, group, time, 1.0)
