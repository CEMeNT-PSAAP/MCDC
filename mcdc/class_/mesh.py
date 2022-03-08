import numpy as np

from mcdc.constant import INF
from mcdc.kernel   import binary_search
from mcdc.print_   import print_error


class Mesh():
    def __init__(self, x=None, y=None, z=None, t=None):
        # Set grid
        if x is None:
            self.x = np.array([-INF,INF])
        else:
            self.x = x
        if y is None:
            self.y = np.array([-INF,INF])
        else:
            self.y = y
        if z is None:
            self.z = np.array([-INF,INF])
        else:
            self.z = z
        if t is None:
            self.t = np.array([-INF,INF])
        else:
            self.t = t

        # Bin sizes
        self.Nt = len(self.t)-1
        self.Nx = len(self.x)-1
        self.Ny = len(self.y)-1
        self.Nz = len(self.z)-1
    
    # TODO: scalar index for tally (tally index would be [g,t,x,y,z])
    def get_index(self, P):
        t = binary_search(P.time, self.t)
        x = binary_search(P.position.x, self.x)
        y = binary_search(P.position.y, self.y)
        z = binary_search(P.position.z, self.z)
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
        idx = binary_search(value, grid)
        if direction > 0.0:
            idx += 1
        dist = (grid[idx] - value)/direction
        return dist
