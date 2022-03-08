import numpy as np

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

class Particle:
    def __init__(self, position, direction, group, time, weight):
        # Particle phase space
        self.position  = position  # cm
        self.direction = direction
        self.group     = group
        self.time      = time # s
        self.weight    = weight

        # Flags
        self.alive = True

        # Tags
        self.cell = None
        self.idx_census_time = None

        # Misc
        self.speed = 1.0 # Updated in particle loop

    def copy(self):
        P = Particle(self.position.copy(), self.direction.copy(), self.group, 
                     self.time, self.weight)

        # Copy tags
        P.cell            = self.cell
        P.idx_census_time = self.idx_census_time
        return P
