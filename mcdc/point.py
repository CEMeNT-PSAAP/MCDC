import numpy as np


# =============================================================================
# Point
# =============================================================================

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    # Addition with another Point
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    def __iadd__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    
    # Multiplication with scalar
    def __mul__(self, other):
        return Point(self.x*other ,self.y*other, self.z*other)
    def __imul__(self, other):
        return Point(self.x*other, self.y*other, self.z*other)

    # Magnitude
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    # Normalize point (important for direction)
    def normalize(self):
        magnitude = self.magnitude()
        self.x    = self.x/magnitude
        self.y    = self.y/magnitude
        self.z    = self.z/magnitude
