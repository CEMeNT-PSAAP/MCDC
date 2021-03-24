from abc           import ABC, abstractmethod
from mcdc.constant import LCG_G, LCG_C, LCG_MOD, LCG_STRIDE


# Global rng
global rng
rng = None # Setup in simulator.py


# =============================================================================
# Random Number Generator
# =============================================================================

class Random(ABC):
    def __init__(self, seed):
        self.seed_master = seed

    @abstractmethod
    def init_history(self, id_):
        pass

    @abstractmethod
    def __call__(self):
        pass


# Linear Congruential Generator
#   [F. B. Brown 1994, "Random Number Generation with Arbitrary Strides]
class RandomLCG(Random):
    def __init__(self, seed=1, g=LCG_G, c=LCG_C, mod=LCG_MOD, stride=LCG_STRIDE):
        Random.__init__(self, seed)
        self.g        = g
        self.c        = c
        self.mod      = mod
        self.mod_mask = mod - 1
        self.stride   = stride
        self.norm     = 1/mod

    def init_history(self, id_):
        n     = id_*self.stride
        g     = self.g
        c     = self.c
        g_new = 1
        c_new = 0

        n = n & self.mod_mask
        while n > 0:
            if n & 1:
                g_new = g_new*g       & self.mod_mask
                c_new = (c_new*g + c) & self.mod_mask

            c = (g+1)*c & self.mod_mask
            g = g*g     & self.mod_mask
            n >>= 1
            
        self.seed = (g_new*self.seed_master + c_new ) & self.mod_mask

    def __call__(self):
        self.seed = (self.g*self.seed + self.c) & self.mod_mask
        return self.seed*self.norm
