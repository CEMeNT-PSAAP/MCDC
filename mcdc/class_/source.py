import numpy as np

from numba              import float64, boolean, config
from numba.experimental import jitclass

import mcdc.kernel as kernel
import mcdc.type_  as type_

from mcdc.class_.particle import Particle
from mcdc.class_.point    import Point, type_point


@jitclass([('flag_box', boolean), ('box', float64[:]), 
           ('position', type_point), ('flag_isotropic', boolean),
           ('direction', type_point), ('energy', float64[:]), 
           ('time', float64[:])])
class Source:
    def __init__(self):
        self.flag_box       = False
        self.box            = np.zeros(6, dtype=np.float64)
        self.position_x     = 0.0
        self.position_y     = 0.0
        self.position_z     = 0.0
        self.flag_isotropic = True
        self.direction_x    = 1.0
        self.direction_y    = 0.0
        self.direction_z    = 0.0
        self.energy         = np.ones(1, dtype=np.float64)
        self.time           = np.zeros(2, dtype=np.float64)
        self.prob           = 1.0
        self.ID             = -1
    
    def get_particle(self, mcdc):
        # Position
        if self.flag_box:
            x = kernel.sample_uniform(self.box[0],self.box[1], mcdc.rng)
            y = kernel.sample_uniform(self.box[2],self.box[3], mcdc.rng)
            z = kernel.sample_uniform(self.box[4],self.box[5], mcdc.rng)
        else:
            x = self.position_x
            y = self.position_y
            z = self.position_z

        # Direction
        if self.flag_isotropic:
            ux, uy, uz = kernel.sample_isotropic_direction(mcdc.rng)
        else:
            ux = self.direction_x
            uy = self.direction_y
            uz = self.direction_z

        # Energy and time
        energy = kernel.sample_discrete(self.energy, mcdc.rng)
        time   = kernel.sample_uniform(self.time[0], self.time[1], mcdc.rng)

        P = type_.make_particle()
        P['x']     = x
        P['y']     = y
        P['z']     = z
        P['ux']    = ux
        P['uy']    = uy
        P['uz']    = uz
        P['group'] = energy
        P['time']  = time

        return P

if not config.DISABLE_JIT:
    type_source = Source.class_type.instance_type
else:
    type_source = None
