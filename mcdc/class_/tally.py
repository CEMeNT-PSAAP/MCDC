from   numba              import int64, float64, boolean, types, njit, config
from   numba.experimental import jitclass
import numpy              as     np

from mcdc.class_.mesh  import Mesh, type_mesh
from mcdc.class_.score import ScoreFlux, ScoreCurrent, ScoreEddington,\
                              type_score_flux, type_score_current,\
                              type_score_eddington

#==============================================================================
# Tally
#==============================================================================
# TODO: implement inheritance/absract jitclass if already supported

@jitclass([('flux', boolean), ('current', boolean), ('eddington', boolean), 
           ('mesh', type_mesh), ('score_flux', type_score_flux),
           ('score_current', type_score_current), 
           ('score_eddington', type_score_eddington)])
class Tally:
    def __init__(self):
        # Score flags
        self.flux      = False
        self.current   = False
        self.eddington = False

        # Uninitialized
        #self.mesh
        #self.score_flux     
        #self.score_current  
        #self.score_eddington

    def allocate_bins(self, Ng, N_iter):
        Nt = self.mesh.Nt
        Nx = self.mesh.Nx
        Ny = self.mesh.Ny
        Nz = self.mesh.Nz

        if self.flux:
            shape = (N_iter, Nt, Ng, Nx, Ny, Nz)
            result_bin = np.zeros(shape, dtype=np.float64)
            self.score_flux = ScoreFlux(result_bin)
        if self.current:
            shape = (N_iter, Nt, Ng, Nx, Ny, Nz, 3)
            result_bin = np.zeros(shape, dtype=np.float64)
            self.score_current = ScoreCurrent(result_bin)
        if self.eddington:
            shape = (N_iter, Nt, Ng, Nx, Ny, Nz, 6)
            result_bin = np.zeros(shape, dtype=np.float64)
            self.score_eddington = ScoreEddington(result_bin)

    def score(self, P, d_move):
        # Get indices
        t, x, y, z = self.mesh.get_index(P)

        # Score
        if self.flux:
            self.score_flux.accumulate(t,x,y,z,d_move,P)
        if self.current:
            self.score_current.accumulate(t,x,y,z,d_move,P)
        if self.eddington:
            self.score_eddington.accumulate(t,x,y,z,d_move,P)
   
    def closeout_history(self):
        if self.flux:
            closeout_history(self.score_flux)
        if self.current:
            closeout_history(self.score_current)
        if self.eddington:
            closeout_history(self.score_eddington)
    
@njit
def closeout_history(score):
    # Accumulate sums of history
    score.sum_   += score.bin_
    score.sum_sq += np.square(score.bin_)

    # Reset bin
    score.bin_.fill(0.0)

if not config.DISABLE_JIT:
    type_tally = Tally.class_type.instance_type
else:
    type_tally = None

#==============================================================================
# Global tally (for eigenvalue mode)
#==============================================================================
# TODO: alpha eigenvalue with delayed neutrons
# TODO: shannon entropy

@jitclass([('k_eff', float64), ('alpha_eff', float64),
           ('nuSigmaF', float64), ('inverse_speed', float64),
           ('k_iterate', float64[:]), ('alpha_iterate', float64[:])])
class TallyGlobal:
    def __init__(self):
        # Effective eigenvalue
        self.k_eff     = 1.0
        self.alpha_eff = 0.0

        # Accumulator
        self.nuSigmaF      = 0.0
        self.inverse_speed = 0.0

    def allocate(self, N_iter):
        # Eigenvalue solution iterate
        self.k_iterate     = np.zeros(N_iter)
        self.alpha_iterate = np.zeros(N_iter)

if not config.DISABLE_JIT:
    type_tally_global = TallyGlobal.class_type.instance_type
else:
    type_tally_global = None
