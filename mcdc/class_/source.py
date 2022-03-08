from abc import ABC, abstractmethod

from mcdc.class_.distribution import DistDelta, DistPointIsotropic, DistPoint
from mcdc.class_.particle     import Particle

class Source(ABC):
    def __init__(self, prob):
        self.prob = prob

    @abstractmethod
    def get_particle(self):
        pass

class SourceSimple(Source):
    def __init__(self, position=None, direction=None, group=None, time=None,
                 prob=1.0):

        Source.__init__(self, prob)
        if position is None:
            self.position = DistPoint()
        else:
            self.position = position
        if direction is None:
            self.direction = DistPointIsotropic()
        else:
            self.direction = direction
        if group is None:
            self.group = DistDelta(0)
        else:
            self.group = group

        if time is None:
            self.time = DistDelta(0.0)
        else:
            self.time = time

    def get_particle(self):
        position  = self.position.sample()
        direction = self.direction.sample()
        group     = self.group.sample()
        time      = self.time.sample()
        direction.normalize()
        return Particle(position, direction, group, time, 1.0)
