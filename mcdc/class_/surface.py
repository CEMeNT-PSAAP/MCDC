# TODO: quadric surface
# TODO: implement inheritance/absract jitclass if already supported

from numba              import int64, float64, boolean, config
from numba.experimental import jitclass

from mcdc.class_.point import Point, type_point
from mcdc.constant     import INF

@jitclass([('ID', int64), ('vacuum', float64), ('reflective', boolean),
          ('A', float64), ('B', float64), ('C', float64), ('D', float64),
          ('n', type_point)])
class Surface:
    """
    General surface: Ax + By + Cz + D = 0
    """

    def __init__(self, ID, vacuum, reflective, A, B, C, D):
        self.ID         = ID
        self.vacuum     = vacuum
        self.reflective = reflective
        self.A          = A
        self.B          = B
        self.C          = C
        self.D          = D
       
        # Calculate normal vector
        self.n = Point(A, B, C)
        self.n.normalize()

    def evaluate(self, P):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        x = P.position.x
        y = P.position.y
        z = P.position.z
        return A*x + B*y + C*z + D

    def distance(self, P):
        A  = self.A
        B  = self.B
        C  = self.C
        D  = self.D
        x  = P.position.x
        y  = P.position.y
        z  = P.position.z
        
        # Calculate distance
        dist = -(A*x + B*y + C*z + D)/self.normal_component(P)

        # Check if particle moves away from the surface
        if dist < 0.0: return INF
        else:          return dist

    def normal_component(self, P):
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        nx = self.n.x
        ny = self.n.y
        nz = self.n.z
        return nx*ux + ny*uy + nz*uz

    def apply_bc(self, P):
        if self.vacuum:
            P.alive = False
        elif self.reflective:
            self.reflect(P)

    def reflect(self, P):
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        nx = self.n.x
        ny = self.n.y
        nz = self.n.z
        c  = 2.0*self.normal_component(P)

        P.direction.x = ux - c*nx
        P.direction.y = uy - c*ny
        P.direction.z = uz - c*nz

if not config.DISABLE_JIT:
    type_surface = Surface.class_type.instance_type
else:
    type_surface = None

# User's python interface
class SurfaceHandle:
    def __init__(self, surface):
        self.surface = surface
    def __pos__(self):
        return [self.surface,+1.0]
    def __neg__(self):
        return [self.surface,-1.0]
