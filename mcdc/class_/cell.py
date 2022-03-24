from numba              import int64, float64, boolean, types, config
from numba.experimental import jitclass

from mcdc.class_.material import type_material
from mcdc.class_.surface  import type_surface

@jitclass([('ID', int64), ('surfaces', types.ListType(type_surface)),
           ('senses', float64[:]), ('material', type_material), 
           ('n_surfaces', int64)])
class Cell:
    def __init__(self, ID, surfaces, senses, material):
        self.ID         = ID
        self.surfaces   = surfaces
        self.senses     = senses
        self.material   = material
        self.n_surfaces = len(surfaces)
    
    # Test if position pos is inside the cell
    def test_point(self, P):
        for i in range(self.n_surfaces):
            if self.surfaces[i].evaluate(P) * self.senses[i] < 0.0:
                return False
        return True

if not config.DISABLE_JIT:
    type_cell = Cell.class_type.instance_type
else:
    type_cell = None
