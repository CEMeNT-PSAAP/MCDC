import numpy as np

from abc import ABC, abstractmethod

import mcdc.mpi

from mcdc.misc     import binary_search
from mcdc.constant import EPSILON, INF


def getidx_energy(val, grid):
    return binary_search(val, grid) + 1

class Tally:
    def __init__(self, name, scores,
                 x=None, y=None, z=None, t=None, 
                 g=None, g_low=None, g_up=None,
                 cells=None, surfaces=None):

        self.name = name

        # Get estimator type (track/speed/cross)
        
        self.scores_track = []
        self.scores_cross = []
        self.scores_speed = []

        # Indexers
        self.getidx_space  = None
        self.getidx_time   = None
        self.getidx_energy = getidx_energy

        # Cartesian space
        if x is not None or y is not None or z is not None:
