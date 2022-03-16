from   numba              import float64
from   numba.experimental import jitclass
import numpy              as     np

spec = [('x', float64), ('y', float64), ('z', float64)]
@jitclass(spec)
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    # Magnitude
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    # Normalize point (important for direction)
    def normalize(self):
        magnitude = self.magnitude()
        self.x    = self.x/magnitude
        self.y    = self.y/magnitude
        self.z    = self.z/magnitude

    # Create copy
    def copy(self):
        return Point(self.x, self.y, self.z)
