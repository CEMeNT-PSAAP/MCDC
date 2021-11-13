from abc import ABC, abstractmethod

from mcdc.point import Point


# =============================================================================
# Particle
# =============================================================================

class Particle:
    def __init__(self, pos, dir, g, time, wgt, cell, census_idx):
        # Particle state
        self.pos   = pos  # cm
        self.dir   = dir
        self.g     = g
        self.time  = time # s
        self.wgt   = wgt
        self.cell  = cell
        
        self.alive = True
        self.speed = None # Determined at particle move
               
        # Previous state (pre particle loop, for tally)
        self.pos_old  = Point(pos.x, pos.y, pos.z)
        self.dir_old  = Point(dir.x, dir.y, dir.z)
        self.g_old    = g
        self.time_old = time
        self.wgt_old  = wgt
        self.cell_old = cell
        
        self.wgt_post = None # Post-collision

        # Census
        self.census_idx = census_idx

        # Particle loop records
        self.distance         = 0.0   # distance traveled during a loop
        self.surface          = None  # surface object hit

    def save_previous_state(self):
        self.pos_old.copy(self.pos)
        self.dir_old.copy(self.dir)
        self.g_old    = self.g
        self.time_old = self.time
        self.wgt_old  = self.wgt
        self.cell_old = self.cell
        
    def reset_record(self):
        self.distance         = 0.0
        self.surface          = None

        # Post collision weight modification?
        if self.wgt_post:
            self.wgt      = self.wgt_post
            self.wgt_post = None

    def create_copy(self):
        return Particle(self.pos, self.dir, self.g, self.time, self.wgt, 
                        self.cell, self.census_idx)
        
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
    def __init__(self, pos, dir, g, time, prob=1.0, wgt=1.0, cell=None, 
                 census_idx=None):
        Source.__init__(self, prob)
        self.pos  = pos
        self.dir  = dir
        self.g    = g
        self.time = time
        self.wgt  = wgt
        self.cell = cell
        self.census_idx = census_idx

    def get_particle(self):
        pos  = self.pos.sample()
        dir  = self.dir.sample()
        g    = self.g.sample()
        time = self.time.sample()
        wgt  = self.wgt
        cell = self.cell
        census_idx = self.census_idx
        dir.normalize()
        return Particle(pos, dir, g, time, wgt, cell, census_idx)
