# TODO: implement inheritance/absract jitclass if already supported

import numpy              as     np
from   numba              import int64, float64, boolean, config
from   numba.experimental import jitclass

from mcdc.class_.point import Point, type_point
from mcdc.constant     import INF

@jitclass([('ID', int64), ('vacuum', float64), ('reflective', boolean),
          ('A', float64), ('B', float64), ('C', float64), ('D', float64),
          ('E', float64), ('F', float64), ('G', float64), ('H', float64),
          ('I', float64), ('J', float64), ('linear', boolean), ('n', type_point)])
class Surface:
    """
    Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    """

    def __init__(self, ID, vacuum, reflective, A, B, C, D, E, F, G, H, I, J):
        self.ID         = ID
        self.vacuum     = vacuum
        self.reflective = reflective
        self.A          = A
        self.B          = B
        self.C          = C
        self.D          = D
        self.E          = E
        self.F          = F
        self.G          = G
        self.H          = H
        self.I          = I
        self.J          = J
        
        # Linear surface
        if A == 0 and B == 0 and C == 0 and D == 0 and E == 0 and F == 0:
            self.linear = True
            self.n      = Point(G,H,I)
            self.n.normalize()
            
    def evaluate(self, P):
        x = P['position']['x']
        y = P['position']['y']
        z = P['position']['z']
        
        G = self.G
        H = self.H
        I = self.I
        J = self.J

        result = G*x + H*y + I*z + J
        
        if self.linear:
            return result

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        F = self.F
        
        return result + A*x*x + B*y*y + C*z*z + D*x*y + E*x*z + F*y*z              

    def normal(self, P):
        if self.linear:
            return self.n
        
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        F = self.F
        G = self.G
        H = self.H
        I = self.I
        x = P['position']['x']
        y = P['position']['y']
        z = P['position']['z']
        
        dx = 2*A*x + D*y + E*z + G
        dy = 2*B*y + D*x + F*z + H
        dz = 2*C*z + E*x + F*y + I
        
        n = Point(dx,dy,dz)
        n.normalize()
        return n
        
    def normal_component(self, P):
        ux = P['direction']['x']
        uy = P['direction']['y']
        uz = P['direction']['z']
        n  = self.normal(P)        
        return n.x*ux + n.y*uy + n.z*uz

    def apply_bc(self, P):
        if self.vacuum:
            P['alive'] = False
        elif self.reflective:
            self.reflect(P)

    def reflect(self, P):
        ux = P['direction']['x']
        uy = P['direction']['y']
        uz = P['direction']['z']
        n  = self.normal(P)
        c  = 2.0*(n.x*ux + n.y*uy + n.z*uz) # 2.0*self.normal_component(P)
                                            # to avoid repeating normalization
        P['direction']['x'] = ux - c*n.x
        P['direction']['y'] = uy - c*n.y
        P['direction']['z'] = uz - c*n.z
    
    def distance(self, P):
        x  = P['position']['x']
        y  = P['position']['y']
        z  = P['position']['z']
        ux = P['direction']['x']
        uy = P['direction']['y']
        uz = P['direction']['z']

        G  = self.G
        H  = self.H
        I  = self.I

        if self.linear:
            distance = -self.evaluate(P)/(G*ux + H*uy + I*uz)
            # Moving away from the surface
            if distance < 0.0: return INF
            else:              return distance
            
        A  = self.A
        B  = self.B
        C  = self.C
        D  = self.D
        E  = self.E
        F  = self.F

        # Quadratic equation constants
        a = A*ux*ux + B*uy*uy + C*uz*uz + D*ux*uy + E*ux*uz + F*uy*uz
        b = 2*(A*x*ux + B*y*uy + C*z*uz) +\
            D*(x*uy + y*ux) + E*(x*uz + z*ux) + F*(y*uz + z*uy) +\
            G*ux + H*uy + I*uz
        c = self.evaluate(P)
        
        determinant = b*b - 4.0*a*c
        
        # Roots are complex  : no intersection
        # Roots are identical: tangent
        # ==> return huge number
        if determinant <= 0.0:
            return INF
        else:
            # Get the roots
            denom = 2.0*a
            sqrt  = np.sqrt(determinant)
            root_1 = (-b + sqrt)/denom
            root_2 = (-b - sqrt)/denom
            
            # Negative roots, moving away from the surface
            if root_1 < 0.0: root_1 = INF
            if root_2 < 0.0: root_2 = INF
            
            # Return the smaller root
            return min(root_1, root_2)
            
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
