from   numba              import float64, config
from   numba.experimental import jitclass
import numpy              as     np

@jitclass([('x', float64), ('y', float64), ('z', float64)])
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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

    # Print
    def print_(self):
        print("(", self.x, self.y, self.z, ")")

if not config.DISABLE_JIT:
    type_point = Point.class_type.instance_type
else:
    type_point = None
