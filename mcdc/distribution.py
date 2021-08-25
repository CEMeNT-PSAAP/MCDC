import numpy as np

from abc import ABC, abstractmethod

import mcdc.random

from mcdc.constant import PI
from mcdc.point    import Point
from math          import floor
from mcdc.misc     import binary_search


class Distribution(ABC):
    """Abstract class for random sampling of a distribution"""
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
        xi = mcdc.random.rng()
        return self.a + xi * (self.b - self.a)

class DistUniformInt(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self):
        xi = mcdc.random.rng()
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
        xi  = mcdc.random.rng()
        return binary_search(xi, self.cdf)

class DistPoint(Distribution):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def sample(self):
        return Point(self.x.sample(), self.y.sample(), self.z.sample())
    
class DistPointIsotropic(Distribution):
    def sample(self):
        # Sample polar cosine and azimuthal angle uniformly
        mu  = 2.0*mcdc.random.rng() - 1.0
        azi = 2.0*PI*mcdc.random.rng()
	
        # Convert to Cartesian coordinates
        c = (1.0 - mu**2)**0.5
        y = np.cos(azi)*c
        z = np.sin(azi)*c
        x = mu
        return Point(x, y, z)

class DistPointCylinderZ(Distribution):
    def __init__(self, x0, y0, radius, bottom, top):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.bottom = bottom
        self.top    = top

    def sample(self):
        xi1 = mcdc.random.rng()
        xi2 = mcdc.random.rng()
        xi3 = mcdc.random.rng()
        r     = self.radius*np.sqrt(xi1)
        theta = 2.0*PI*xi2
        x     = self.x0 + r*np.cos(theta)
        y     = self.y0 + r*np.sin(theta)
        z     = self.bottom + xi3 * (self.top - self.bottom)
        return Point(x,y,z)
