from   numba              import int64, float64, boolean, config
from numba              import types
from numba.experimental import jitclass
from numba.typed        import List

from mcdc.class_.point   import type_point
from mcdc.class_.cell    import type_cell
from mcdc.class_.surface import type_surface

@jitclass([('position', type_point), ('direction', type_point),
           ('group', int64), ('time', float64), ('weight', float64),
           ('alive', boolean), ('speed', float64), 
           ('cell', type_cell), ('surface', type_surface)])
class Particle:
    def __init__(self, position, direction, group, time, weight):
        self.position  = position # cm
        self.direction = direction
        self.group     = group
        self.time      = time # s
        self.weight    = weight
        self.alive     = True
        self.speed     = 1.0 # Updated in particle loop

        # Uninitialized
        # self.cell
        # self.surface

    def copy(self):
        P = Particle(self.position.copy(), self.direction.copy(), self.group, 
                     self.time, self.weight)
        P.cell = self.cell
        return P

if not config.DISABLE_JIT:
    type_particle = Particle.class_type.instance_type
else:
    type_particle = None

@jitclass([('source', types.ListType(type_particle)),
           ('history', types.ListType(type_particle)),
           ('fission', types.ListType(type_particle)),
           ('stored', types.ListType(type_particle))])
class Bank:
    def __init__(self):
        self.source  = List.empty_list(type_particle)
        self.history = List.empty_list(type_particle)
        self.fission = self.history
        self.stored  = List.empty_list(type_particle)

if not config.DISABLE_JIT:
    type_bank = Bank.class_type.instance_type
else:
    type_bank = None
