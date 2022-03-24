# TODO: implement inheritance/absract jitclass if already supported

from   numba              import int64, float64, boolean, njit, config
from   numba.experimental import jitclass
import numpy              as     np

type_bin  = float64[:,:,:,:,:]
type_mean = float64[:,:,:,:,:,:]
spec_score = [('bin_', type_bin), 
              ('sum_', type_bin),
              ('sum_sq', type_bin),
              ('mean', type_mean),
              ('sdev', type_mean)]

@jitclass(spec_score)
class ScoreFlux:
    def __init__(self, shape):
        self.bin_, self.sum_, self.sum_sq, self.mean, self.sdev = set_bins(shape)

    def accumulate(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        self.bin_[t, P.group, x, y, z] += flux

# Add additional dimension for current and eddington
type_bin  = float64[:,:,:,:,:,:]
type_mean = float64[:,:,:,:,:,:,:]
spec_score = [('bin_', type_bin), 
              ('sum_', type_bin),
              ('sum_sq', type_bin),
              ('mean', type_mean),
              ('sdev', type_mean)]

@jitclass(spec_score)
class ScoreCurrent:
    def __init__(self, shape):
        self.bin_, self.sum_, self.sum_sq, self.mean, self.sdev = set_bins(shape)

    def accumulate(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        self.bin_[t, P.group, x, y, z, 0] += flux*P.direction.x
        self.bin_[t, P.group, x, y, z, 1] += flux*P.direction.y
        self.bin_[t, P.group, x, y, z, 2] += flux*P.direction.z

@jitclass(spec_score)
class ScoreEddington:
    def __init__(self, shape):
        self.bin_, self.sum_, self.sum_sq, self.mean, self.sdev = set_bins(shape)

    def accumulate(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        self.bin_[t, P.group, x, y, z, 0] += flux*ux*ux
        self.bin_[t, P.group, x, y, z, 1] += flux*ux*uy
        self.bin_[t, P.group, x, y, z, 2] += flux*ux*uz
        self.bin_[t, P.group, x, y, z, 3] += flux*uy*uy
        self.bin_[t, P.group, x, y, z, 4] += flux*uy*uz
        self.bin_[t, P.group, x, y, z, 5] += flux*uz*uz

if not config.DISABLE_JIT:
    type_score_flux      = ScoreFlux.class_type.instance_type
    type_score_current   = ScoreCurrent.class_type.instance_type
    type_score_eddington = ScoreEddington.class_type.instance_type
else:
    type_score_flux      = None
    type_score_current   = None
    type_score_eddington = None

@njit
def set_bins(mean):
    # Results
    mean = mean
    sdev = np.zeros_like(mean)

    # History accumulator
    shape_reduced = mean.shape[1:]
    bin_ = np.zeros(shape_reduced, dtype=np.float64) # Skip the N_iter

    # Sums of history
    sum_   = np.zeros_like(bin_)
    sum_sq = np.zeros_like(bin_)
    
    return bin_, sum_, sum_sq, mean, sdev

