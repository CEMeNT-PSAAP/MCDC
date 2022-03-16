from   mcdc.class_.particle import Particle
from   mcdc.class_.point    import Point
import mcdc.kernel          as     kernel

# TODO: wait until Numba jitclass supports inheritance/absract class

class SourcePointIsotropic:
    def __init__(self, position, energy, time):
        self.position = position
        self.energy   = energy
        self.time     = time
    
    def get_particle(self):
        position  = self.position.copy()
        direction = kernel.sample_isotropic_direction()
        energy    = kernel.sample_discrete(self.energy)
        time      = kernel.sample_uniform(self.time[0], self.time[1])
        return Particle(position, direction, energy, time, 1.0)

class SourcePointMono:
    def __init__(self, position, direction, energy, time):
        self.position  = position
        self.direction = direction
        self.energy    = energy
        self.time      = time
    
    def get_particle(self):
        position  = self.position.copy()
        direction = self.direction.copy()
        energy    = kernel.sample_discrete(self.energy)
        time      = kernel.sample_uniform(self.time[0], self.time[1])
        return Particle(position, direction, energy, time, 1.0)

class SourceBoxIsotropic:
    def __init__(self, box, energy, time):
        self.box    = box
        self.energy = energy
        self.time   = time
    
    def get_particle(self):
        x = kernel.sample_uniform(self.box[0,0],self.box[0,1])
        y = kernel.sample_uniform(self.box[1,0],self.box[1,1])
        z = kernel.sample_uniform(self.box[2,0],self.box[2,1])
        position  = Point(x,y,z)
        direction = kernel.sample_isotropic_direction()
        energy    = kernel.sample_discrete(self.energy)
        time      = kernel.sample_uniform(self.time[0], self.time[1])
        return Particle(position, direction, energy, time, 1.0)

class SourceBoxMono:
    def __init__(self, box, direction, energy, time):
        self.box       = box
        self.direction = direction
        self.energy    = energy
        self.time      = time
    
    def get_particle(self):
        x = kernel.sample_uniform(self.box[0,0],self.box[0,1])
        y = kernel.sample_uniform(self.box[1,0],self.box[1,1])
        z = kernel.sample_uniform(self.box[2,0],self.box[2,1])
        position  = Point(x,y,z)
        direction = self.direction.copy()
        energy    = kernel.sample_discrete(self.energy)
        time      = kernel.sample_uniform(self.time[0], self.time[1])
        return Particle(position, direction, energy, time, 1.0)
