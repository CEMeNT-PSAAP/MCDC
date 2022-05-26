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
        set_bins(self, shape)

    def accumulate(self, g, t, x, y, z, flux, P):
        self.bin_[g, t, x, y, z] += flux

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
        set_bins(self, shape)

    def accumulate(self, g, t, x, y, z, flux, P):
        self.bin_[g, t, x, y, z, 0] += flux*P['ux']
        self.bin_[g, t, x, y, z, 1] += flux*P['uy']
        self.bin_[g, t, x, y, z, 2] += flux*P['uz']

@jitclass(spec_score)
class ScoreEddington:
    def __init__(self, shape):
        set_bins(self, shape)

    def accumulate(self, g, t, x, y, z, flux, P):
        ux = P['ux']
        uy = P['uy']
        uz = P['uz']
        self.bin_[g, t, x, y, z, 0] += flux*ux*ux
        self.bin_[g, t, x, y, z, 1] += flux*ux*uy
        self.bin_[g, t, x, y, z, 2] += flux*ux*uz
        self.bin_[g, t, x, y, z, 3] += flux*uy*uy
        self.bin_[g, t, x, y, z, 4] += flux*uy*uz
        self.bin_[g, t, x, y, z, 5] += flux*uz*uz

if not config.DISABLE_JIT:
    type_score_flux      = ScoreFlux.class_type.instance_type
    type_score_current   = ScoreCurrent.class_type.instance_type
    type_score_eddington = ScoreEddington.class_type.instance_type
else:
    type_score_flux      = None
    type_score_current   = None
    type_score_eddington = None

@njit
def set_bins(self, shape):
    # Results
    self.mean = np.zeros(shape, dtype=np.float64)
    self.sdev = np.zeros_like(self.mean)

    # History accumulator
    shape_reduced = shape[1:]
    self.bin_ = np.zeros(shape_reduced, dtype=np.float64) # Skip the N_iter

    # Sums of history
    self.sum_   = np.zeros_like(self.bin_)
    self.sum_sq = np.zeros_like(self.bin_)
