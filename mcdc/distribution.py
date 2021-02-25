import numpy as np

import rng
from particle import Point



class DistDelta:
    def __init__(self, value):
        self.value = value
    def sample(self):
        return self.value

class DistUniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self):
        xi = rng.uniform()
        return self.a + xi * (self.b - self.a)

class DistPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def sample(self):
        return Point(self.x.sample(), self.y.sample(), self.z.sample())
    
class DistPointIsotropic:
    def sample(self):
        # Sample polar cosine and azimuthal angle uniformly
        mu  = 2.0*rng.uniform() - 1.0
        azi = 2.0*np.pi*rng.uniform()
	
        # Convert to Cartesian coordinates
        c = (1.0 - mu**2)**0.5
        y = np.cos(azi)*c
        z = np.sin(azi)*c
        x = mu
        return Point(x, y, z)