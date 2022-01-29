from abc import ABC, abstractmethod

import numpy as np
import itertools

from mcdc.constant import INF


# =============================================================================
# Surface
# =============================================================================

# Abstract base class
class Surface(ABC):
    """
    Abstract class for geometry surface

    ...

    Attributes
    ----------
    type : str
        Surface type
    id : int
        Surface id
    name : str
        Surface name
    bc : BC
        Surface boundary condition implementation (ransmission, vacuum,
        or reflective) which is described in the subclass `BC`

    Abstract Methods
    ----------------
    evaluate(pos)
        Evaluate if position `pos` is on the + or - side of the surface
    distance(pos, dir)
        Return the distance for a ray with position `pos` and direction
        `dir` to hit the surface
    normal(po, dir)
        Return dot product of the surface normal+ and a ray at position
        `pos` with direction `dir`. This is used for some surface 
        tallies
    """

    _ids = itertools.count(0)

    def __init__(self, type_, bc, name):
        self.type = type_
        self.id   = next(self._ids)
        self.name = name
        
        # Set BC if transmission or vacuum
        if bc == "transmission":
            self.bc = self.BCTransmission()
        elif bc == "vacuum":
            self.bc = self.BCVacuum()
        else:
            self.bc = None # reflective and white are defined in child class

    # Bounding surfaces
    def __pos__(self):
        return [self,+1]
    def __neg__(self):
        return [self,-1]

    # Abstract methods
    @abstractmethod
    def evaluate(self, pos, t):
        pass
    @abstractmethod
    def distance(self, pos, dir, t, speed):
        pass
    @abstractmethod
    def normal(self, pos, dir, t):
        pass

    # =========================================================================
    # Boundary Conditions
    # =========================================================================

    class BC(ABC):
        """
        Abstract subclass for surface BC implementations

        `BCTransmission` and `BCVacuum` are identical for all surface
        types. `BCReflective` is dependent on the surface type.
        """

        def __init__(self, type_):
            self.type = type_        
        @abstractmethod
        def __call__(self, P):
            pass
            
    class BCTransmission(BC):
        def __init__(self):
            Surface.BC.__init__(self, "transmission")
        def __call__(self, P):
            pass

    class BCVacuum(BC):
        def __init__(self):
            Surface.BC.__init__(self, "vacuum")
        def __call__(self, P):
            P.alive = False
    
    
# =============================================================================
# Axis-aligned Plane Surfaces
# =============================================================================

class SurfacePlaneX(Surface):
    def __init__(self, x0, bc='transmission', name=None):
        Surface.__init__(self, "PlaneX", bc, name)
        self.x0 = x0
        
        # Set BC if reflective
        if bc == "reflective":
            self.bc = self.BCReflective()
        
    def evaluate(self, pos, t):
        return pos.x - self.x0

    def distance(self, pos, dir, t, speed):
        x  = pos.x
        ux = dir.x
        
        # Calculate distance
        dist = (self.x0 - x)/ux
        
        # Check if particle moves away from the surface
        if dist < 0.0: return INF
        else:          return dist

    def normal(self, pos, dir, t):
        return dir.x
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            P.dir.x *= -1

class MovingSurfacePlaneX(SurfacePlaneX):
    def __init__(self, x0, v, bc='transmission', name=None):
        Surface.__init__(self, "MovingPlaneX", bc, name)
        self.x0 = x0
        self.v  = v
        
        # Set BC if reflective
        if bc == "reflective":
            self.bc = self.BCReflective()
        
    def evaluate(self, pos, t):
        x0 = self.x0 + t*self.v
        return pos.x - x0

    def distance(self, pos, dir, t, speed):
        # 1: particle
        # 2: surface

        x1 = pos.x
        ux = dir.x
        v1 = speed

        x2 = self.x0 + t*self.v
        v2 = self.v

        # Calculate distance
        dist = v1*(x2 - x1)/(v1*ux-v2)
        
        # Check if particle moves away from the surface
        if dist < 0.0: return INF
        else:          return dist

    def normal(self, pos, dir, t):
        return dir.x
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            P.dir.x *= -1

'''
class SurfacePlaneZ(Surface):
    def __init__(self, z0, bc='transmission', name=None):
        Surface.__init__(self, "PlaneZ", bc, name)
        self.z0 = z0
        
        # Set BC if reflective
        if bc == "reflective":
            self.bc = self.BCReflective()
        
    def evaluate(self, pos):
        return pos.z - self.z0

    def distance(self, pos, dir):
        z  = pos.z
        uz = dir.z
        
        # Calculate distance
        dist = (self.z0 - z)/uz
        
        # Check if particle moves away from the surface
        if dist < 0.0: return INF
        else:          return dist

    def normal(self, pos, dir):
        return dir.z
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            P.dir.z *= -1


# =============================================================================
# Axis-aligned Infinite Cylinder Surfaces
# =============================================================================

class SurfaceCylinderZ(Surface):
    def __init__(self, x0, y0, R, bc='transmission', name=None):
        Surface.__init__(self, "CylinderZ", bc, name)
        self.x0   = x0
        self.y0   = y0
        self.R_sq = R*R
        
        # Set BC if reflective
        if bc == "reflective":
            self.bc = self.BCReflective()
        
    def evaluate(self, pos):
        x = pos.x - self.x0
        y = pos.y - self.y0
        return x*x + y*y - self.R_sq

    def distance(self, pos, dir):
        x = pos.x - self.x0
        y = pos.y - self.y0

        # Set quadratic constants
        a = 1.0 - dir.z*dir.z
        b = 2*(x*dir.x + y*dir.y)
        c = x*x + y*y - self.R_sq

        # Calculate determinant
        D = b*b - 4*a*c

        # Return INF if tangent (D=0) or imaginer roots (D<0)
        if D <= 0:
            return INF
        
        # Get the roots
        D_sqrt = np.sqrt(D)
        root1  = (-b + D_sqrt)/(2*a)
        root2  = (-b - D_sqrt)/(2*a)

        # Return INF if moving away from surface (root < 0)
        if root1 < 0: root1 = INF
        if root2 < 0: root2 = INF

        # Return the smallest root
        return min(root1, root2)

    def normal(self, pos, dir):
        # TODO
        return 1.0
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            # TODO
            pass


    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            # TODO
            pass

'''
# =============================================================================
# Cell
# =============================================================================

class Cell:
    """
    Geometry cell

    A cell is defined by its bounding surfaces and filling material

    Attributes
    ----------
    surfaces : list of mcdc.Surface
        The bounding surfaces
    senses : list of int
        The corresponding sense (+/-) of the surface with the same index
    material : mcdc.Material
        The `mcdc.Material` object that fills the cell

    Methods
    -------
    test_point(pos)
        Test if position `pos` is inside the cell
    """

    def __init__(self, bounds, material):
        """
        Arguments
        ---------
        bounds : list of list([mcdc.Surface, int])
            The first column is the `mcdc.Surface` objects bounding the cell.
            The second column is the corresponding sense (+/-).
        material : mcdc.Material
            The `mcdc.Material` object that fills the cell
        """

        self.n_surfaces = len(bounds)
        self.surfaces   = []
        self.senses     = []
        for i in range(self.n_surfaces):
            self.surfaces.append(bounds[i][0])
            self.senses.append(bounds[i][1])
        self.material = material

    # Test if position pos is inside the cell
    def test_point(self, pos, t):
        for i in range(self.n_surfaces):
            if self.surfaces[i].evaluate(pos,t) * self.senses[i] < 0:
                return False
        return True
