import numpy as np

from math import floor

import mcdc.random

from mcdc.constant import INF
from mcdc.print    import print_error
from mcdc.misc     import binary_search

import matplotlib.pyplot as plt

class WeightWindow:
    def __init__(self, x=None, y=None, z=None, t=None, window=None):
        ax_expand = []

        if t is None:
            self.t = np.array([-INF,INF])
            ax_expand.append(0)
        else:
            self.t = t

        if x is None:
            self.x = np.array([-INF,INF])
            ax_expand.append(1)
        else:
            self.x = x
        
        if y is None:
            self.y = np.array([-INF,INF])
            ax_expand.append(2)
        else:
            self.y = y
        
        if z is None:
            self.z = np.array([-INF,INF])
            ax_expand.append(3)
        else:
            self.z = z
        
        if window is None:
            dim = [len(self.t)-1, len(self.x)-1, len(self.y)-1, len(self.z)-1]
            self.window = np.ones(dim)
        else:
            self.window = window

            for ax in ax_expand:
                self.window = np.expand_dims(self.window, axis=ax)

        self.window /= np.max(self.window)

    def __call__(self, P, bank):
        # Get index
        x = binary_search(P.pos.x, self.x)
        y = binary_search(P.pos.y, self.y)
        z = binary_search(P.pos.z, self.z)
        t = binary_search(P.time, self.t)

        # Weight target
        w_target = self.window[t,x,y,z]
       
        # Surviving probability
        p = P.wgt/w_target

        # If above target
        if p > 1.0:
            # Keep the original particle
            P.wgt = w_target

            # Splitting
            n_split = floor(p)
            for i in range(n_split-1):
                bank.append(P.create_copy())

            # Russian roulette
            p -= n_split
            xi = mcdc.random.rng()
            if xi < p:
                bank.append(P.create_copy())
        # Below target
        else:
            # Russian roulette
            xi = mcdc.random.rng()
            if xi > p:
                P.alive = False
