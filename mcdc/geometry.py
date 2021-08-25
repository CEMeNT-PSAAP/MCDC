from abc import ABC, abstractmethod

import numpy as np

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

    def __init__(self, type_, bc, id_, name):
        self.type = type_
        self.id   = id_
        self.name = name
        
        # Set BC if transmission or vacuum
        if bc == "transmission":
            self.bc = self.BCTransmission()
        elif bc == "vacuum":
            self.bc = self.BCVacuum()
        else:
            self.bc = None # reflective and white are defined in child class

    # Abstract methods
    @abstractmethod
    def evaluate(self, pos):
        pass
    @abstractmethod
    def distance(self, pos, dir):
        pass
    @abstractmethod
    def normal(self, pos, dir):
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
    def __init__(self, x0, bc='transmission', id_=None, name=None):
        Surface.__init__(self, "PlaneX", bc, id_, name)
        self.x0 = x0
        
        # Set BC if reflective
        if bc == "reflective":
            self.bc = self.BCReflective()
        
    def evaluate(self, pos):
        return pos.x - self.x0

    def distance(self, pos, dir):
        x  = pos.x
        ux = dir.x
        
        # Calculate distance
        dist = (self.x0 - x)/ux
        
        # Check if particle moves away from the surface
        if dist < 0.0: return INF
        else:          return dist

    def normal(self, pos, dir):
        return dir.x
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BCReflective(Surface.BC):
        def __init__(self):
            Surface.BC.__init__(self, "reflective")
        def __call__(self, P):
            P.dir.x *= -1

class SurfacePlaneZ(Surface):
    def __init__(self, z0, bc='transmission', id_=None, name=None):
        Surface.__init__(self, "PlaneZ", bc, id_, name)
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
    def __init__(self, x0, y0, R, bc='transmission', id_=None, name=None):
        Surface.__init__(self, "CylinderZ", bc, id_, name)
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
# =============================================================================
# Cell
# =============================================================================

class Cell:
    """
    Class for geometry cell

    ...

    Attributes
    ----------
    id : int
        Cell id
    name : str
        Cell name
    name : str
        Cell name
    surfaces : list [Surface, int]
        A 2xN list. The first row is the `Surface` objects bounding the
        cell. The second row is the sense (+/-) of the corresponding
        bounding surfaces.
    material : Material
        The `Material` object that fills the cell

    Methods
    ----------------
    test_point(pos)
        Test if position `pos` is inside the cell
    """

    def __init__(self, surfaces, material, id_=None, name=None):
        self.id       = id_
        self.name     = name
        self.surfaces = surfaces # 0: surface, 1: sense
        self.material = material

    # Test if position pos is inside the cell
    def test_point(self, pos):
        for surface in self.surfaces:
            if surface[0].evaluate(pos) * surface[1] < 0:
                return False
        return True
