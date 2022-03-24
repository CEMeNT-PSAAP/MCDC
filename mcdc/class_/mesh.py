from   numba              import float64, int64, config
from   numba.experimental import jitclass
import numpy              as     np

from   mcdc.constant import INF
import mcdc.kernel   as     kernel

@jitclass([('x', float64[:]), ('y', float64[:]), ('z', float64[:]),
           ('t', float64[:]), ('Nt', int64), ('Nx', int64), ('Ny', int64), 
           ('Nz', int64)])
class Mesh():
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

        # Bin sizes
        self.Nt = len(self.t)-1
        self.Nx = len(self.x)-1
        self.Ny = len(self.y)-1
        self.Nz = len(self.z)-1
    
    # TODO: scalar index for tally (tally index would be [g,t,x,y,z])
    def get_index(self, P):
        t = kernel.binary_search(P.time, self.t)
        x = kernel.binary_search(P.position.x, self.x)
        y = kernel.binary_search(P.position.y, self.y)
        z = kernel.binary_search(P.position.z, self.z)
        return t, x, y, z

    # TODO: add mesh indices to Particle
    def distance(self, P):
        x = P.position.x
        y = P.position.y
        z = P.position.z
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        t = P.time
        v = P.speed

        d = INF
        d = min(d, self._distance_search(x, ux, self.x))
        d = min(d, self._distance_search(y, uy, self.y))
        d = min(d, self._distance_search(z, uz, self.z))
        d = min(d, self._distance_search(t, 1.0, self.t))
        return d

    def _distance_search(self, value, direction, grid):
        if direction == 0.0:
            return INF
        idx = kernel.binary_search(value, grid)
        if direction > 0.0:
            idx += 1
        dist = (grid[idx] - value)/direction
        return dist

if not config.DISABLE_JIT:
    type_mesh = Mesh.class_type.instance_type
else:
    type_mesh = None
