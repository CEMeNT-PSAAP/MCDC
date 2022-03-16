from   abc   import ABC, abstractmethod
from   math  import floor
import numpy as     np

from   mcdc.class_.point import Point
from   mcdc.constant     import PI
import mcdc.kernel       as     kernel

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

class Distribution(ABC):
    """Abstract class for sampling of a distribution"""
    @abstractmethod
    def sample(self):
        pass

class DistDelta(Distribution):
    def __init__(self, value):
        self.value = value
    def sample(self):
        return self.value

class DistUniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self):
        xi = mcdc.rng()
        return self.a + xi * (self.b - self.a)

class DistUniformInt(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self):
        xi = mcdc.rng()
        return self.a + floor(xi*(self.b - self.a))

class DistGroup(Distribution):
    def __init__(self, pmf):
        self.G   = len(pmf)
        self.cdf = np.zeros(self.G+1)

        # Normalize pmf
        norm  = sum(pmf)
        pmf  /= norm

        # Create cdf
        for i in range(self.G):
            self.cdf[i+1] = self.cdf[i] + pmf[i]
    def sample(self):
        xi  = mcdc.rng()
        return kernel.binary_search(xi, self.cdf)

class DistPoint(Distribution):
    def __init__(self, x=DistDelta(0.0), y=DistDelta(0.0), z=DistDelta(0.0)):
        self.x = x
        self.y = y
        self.z = z
    def sample(self):
        return Point(self.x.sample(), self.y.sample(), self.z.sample())
    
class DistPointIsotropic(Distribution):
    def sample(self):
        return kernel.isotropic_direction()

class DistPointCylinderZ(Distribution):
    def __init__(self, x0, y0, radius, bottom, top):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.bottom = bottom
        self.top    = top

    def sample(self):
        xi1 = mcdc.rng()
        xi2 = mcdc.rng()
        xi3 = mcdc.rng()
        r     = self.radius*np.sqrt(xi1)
        theta = 2.0*PI*xi2
        x     = self.x0 + r*np.cos(theta)
        y     = self.y0 + r*np.sin(theta)
        z     = self.bottom + xi3 * (self.top - self.bottom)
        return Point(x,y,z)
