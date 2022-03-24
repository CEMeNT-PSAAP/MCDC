from   numba              import int64, config, uint64
from   numba.experimental import jitclass
import numpy              as     np

# Linear Congruential Generator
#   [F. B. Brown 1994, "Random Number Generation with Arbitrary Strides]
@jitclass([('seed_master', int64), ('seed_base', int64), ('seed', int64),
           ('g', int64), ('c', int64), ('mod', uint64), ('stride', int64)])
class Random:
    def __init__(self):
        seed = 1 
        self.seed_master = seed
        self.seed_base   = seed # For skip ahead
        self.seed        = seed
        self.g           = 2806196910506780709
        self.c           = 1
        self.mod         = 2**63
        self.stride      = 152917

    def set_seed(self, seed):
        self.seed_master = seed
        self.seed_base   = seed
        self.seed        = seed

    def set_stride(self, stride):
        self.stride = stride

    def rebase(self):
        self.seed_base = self.seed

    def skip_ahead_strides(self, skip):
        self._skip_ahead(skip, self.stride)

    def skip_ahead(self, skip):
        self._skip_ahead(skip, 1)

    def _skip_ahead(self, skip, stride):
        n        = skip*stride
        g        = self.g
        c        = self.c
        g_new    = 1
        c_new    = 0
        mod      = self.mod
        mod_mask = mod - 1

        n = n & mod_mask
        while n > 0:
            if n & 1:
                g_new = g_new*g       & mod_mask
                c_new = (c_new*g + c) & mod_mask

            c = (g+1)*c & mod_mask
            g = g*g     & mod_mask
            n >>= 1

        self.seed = (g_new*self.seed_base + c_new ) & mod_mask

    def random(self):
        g        = self.g
        c        = self.c
        mod      = self.mod
        mod_mask = mod - 1

        self.seed = (g*self.seed + c) & mod_mask
        return np.float64(self.seed/mod)

if not config.DISABLE_JIT:
    type_random = Random.class_type.instance_type
else:
    type_random = None
