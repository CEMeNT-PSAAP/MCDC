import numpy as np

from constant import small_kick


# =============================================================================
# Surface
# =============================================================================

class Surface:
    def __init__(self, type, BC, ID, name):
        self.type = type
        self.ID   = ID
        self.name = name
        
        # Set BC if transmission or vacuum
        if BC == "transmission":
            self.BC = self.BC_Transmission()
        elif BC == "vacuum":
            self.BC = self.BC_Vacuum()
        else:
            self.BC = None # reflective and white are defined in child class

    # Evaluate if position pos is on the + or - side of the surface
    def evaluate(self, pos):
        pass # Defined in child class
    
    # Distance for a ray with position pos and direction dir to hit the surface
    def distance(self, pos, dir):
        pass # Defined in child class

    # Dot product of surface normal+ and a ray at pos with direction dir
    def normal(self, pos, dir):
        pass # Defined in child class

    # Procedures for particle P hitting the surface
    def hit(self, P):
        # Record surface hit
        P.surface = self
        
        # Implement BC
        self.BC(P)

        # Small kick (see constant.py) to make sure crossing
        #   TODO: Better idea?
        P.pos          += P.dir*small_kick
        P.time         += small_kick/P.speed
        P.distance     += small_kick

    # =========================================================================
    # BC implementations
    # =========================================================================

    class BC_Transmission:
        def __init__(self):
            self.type = "transmission"
        def __call__(self, P):
            pass
    
    class BC_Vacuum:
        def __init__(self):
            self.type = 'vacuum'
        def __call__(self, P):
            P.alive = False
    
    
# =============================================================================
# Surface Axis-parallel Plane
# =============================================================================

class SurfacePlaneX(Surface):
    def __init__(self, x0, BC, ID=None, name=None):
        Surface.__init__(self, "PlaneX", BC, ID, name)
        self.x0 = x0
        
        # Set BC if reflective
        if BC == "reflective":
            self.BC = self.BC_Reflective()
        
    def evaluate(self, pos):
        return pos.x - self.x0

    def distance(self, pos, dir):
        x  = pos.x;
        ux = dir.x;
        
        # Calculate distance
        dist = (self.x0 - x)/ux;
        
        # Check if particle moves away from the surface
        if dist < 0.0: return np.inf
        else:          return dist

    def normal(self, pos, dir):
        return dir.x
    
    # =========================================================================
    # BC implementations
    # =========================================================================

    class BC_Reflective:
        def __init__(self):
            self.type = "reflective"
        def __call__(self, P):
            P.dir.x *= -1


# =============================================================================
# Cell
# =============================================================================

class Cell:
    def __init__(self, surfaces, material, ID=None, name=None):
        self.ID       = ID
        self.name     = name
        self.surfaces = surfaces # 0: surface, 1: sense
        self.material = material

    # Test if position pos is inside the cell
    def test_point(self, pos):
        for surface in self.surfaces:
            if surface[0].evaluate(pos) * surface[1] < 0:
                return False
        return True