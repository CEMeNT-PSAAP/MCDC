from mcdc.point import Point


# =============================================================================
# Particle
# =============================================================================

class Particle:
    def __init__(self, pos, dir, g, time, wgt, cell):
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
        
        # Particle loop records
        self.distance         = 0.0   # distance traveled during a loop
        self.surface          = None  # surface object hit
        self.collision        = False
        self.fission_neutrons = 0     # Number of fission neutrons generated

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
        self.collision        = False
        self.fission_neutrons = 0
        
# =============================================================================
# Source
# =============================================================================

class SourceSimple:
    def __init__(self, pos, dir, g, time, wgt=1.0, cell=None):
        self.pos  = pos
        self.dir  = dir
        self.g    = g
        self.time = time
        self.wgt  = wgt
        self.cell = cell

    def get_particle(self):
        pos  = self.pos.sample()
        dir  = self.dir.sample()
        g    = self.g.sample()
        time = self.time.sample()
        wgt  = self.wgt
        cell = self.cell
        dir.normalize()
        return Particle(pos, dir, g, time, wgt, cell)
