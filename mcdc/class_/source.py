# TODO: implement inheritance/absract jitclass if already supported

from   numba              import float64, boolean, config
from   numba.experimental import jitclass
import numpy              as     np

from   mcdc.class_.particle import Particle
from   mcdc.class_.point    import Point, type_point
import mcdc.kernel          as     kernel

@jitclass([('flag_box', boolean), ('box', float64[:]), 
           ('position', type_point), ('flag_isotropic', boolean),
           ('direction', type_point), ('energy', float64[:]), 
           ('time', float64[:])])
class Source:
    def __init__(self):
        self.flag_box       = False
        self.box            = np.zeros(6, dtype=np.float64)
        self.position       = Point(0.0, 0.0, 0.0)
        self.flag_isotropic = True
        self.direction      = Point(1.0, 0.0, 0.0)
        self.energy         = np.ones(1, dtype=np.float64)
        self.time           = np.zeros(2, dtype=np.float64)
    
    def get_particle(self, mcdc):
        # Position
        if self.flag_box:
            x = kernel.sample_uniform(self.box[0],self.box[1], mcdc.rng)
            y = kernel.sample_uniform(self.box[2],self.box[3], mcdc.rng)
            z = kernel.sample_uniform(self.box[4],self.box[5], mcdc.rng)
            position = Point(x,y,z)
        else:
            position = self.position.copy()

        # Direction
        if self.flag_isotropic:
            direction = kernel.sample_isotropic_direction(mcdc.rng)
        else:
            direction = self.direction.copy()

        # Energy and time
        energy    = kernel.sample_discrete(self.energy, mcdc.rng)
        time      = kernel.sample_uniform(self.time[0], self.time[1], mcdc.rng)

        return Particle(position, direction, energy, time, 1.0)

if not config.DISABLE_JIT:
    type_source = Source.class_type.instance_type
else:
    type_source = None
