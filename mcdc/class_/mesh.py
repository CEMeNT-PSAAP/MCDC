import math
import numpy as np

from numba              import float64, int64, config
from numba.experimental import jitclass

from mcdc.constant import INF

# TODO: non-uniform mesh
'''
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
'''

@jitclass([('x0', float64), ('dx', float64), ('Nx', int64),
           ('y0', float64), ('dy', float64), ('Ny', int64),
           ('z0', float64), ('dz', float64), ('Nz', int64),
           ('t0', float64), ('dt', float64), ('Nt', int64)])
class Mesh():
    def __init__(self, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, t0, dt, Nt):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.t0 = t0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nt = Nt
    
    # TODO: scalar index for tally (tally index would be [g,t,x,y,z])
    def get_index(self, P):
        t = math.floor((P.time       - self.t0)/self.dt)
        x = math.floor((P.position.x - self.x0)/self.dx)
        y = math.floor((P.position.y - self.y0)/self.dy)
        z = math.floor((P.position.z - self.z0)/self.dz)
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
        d = min(d, self._distance_search(x, ux, self.x0, self.dx))
        d = min(d, self._distance_search(y, uy, self.y0, self.dy))
        d = min(d, self._distance_search(z, uz, self.z0, self.dz))
        d = min(d, self._distance_search(t, 1.0/v, self.t0, self.dt))

        return d

    def _distance_search(self, value, direction, x0, dx):
        if direction == 0.0:
            return INF
        idx = math.floor((value - x0)/dx)
        if direction > 0.0:
            idx += 1
        ref = x0 + idx*dx
        dist = (ref - value)/direction
        return dist

    def x(self):
        return np.linspace(self.x0, self.x0 + self.Nx*self.dx, self.Nx+1)
    def y(self):
        return np.linspace(self.y0, self.y0 + self.Ny*self.dy, self.Ny+1)
    def z(self):
        return np.linspace(self.z0, self.z0 + self.Nz*self.dz, self.Nz+1)
    def t(self):
        return np.linspace(self.t0, self.t0 + self.Nt*self.dt, self.Nt+1)

if not config.DISABLE_JIT:
    type_mesh = Mesh.class_type.instance_type
else:
    type_mesh = None
