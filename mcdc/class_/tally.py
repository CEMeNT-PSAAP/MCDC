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
           ('score_eddington', type_score_eddington),
           ('tracklength', boolean), ('crossing', boolean),
           ('crossing_x', boolean), ('crossing_t', boolean), 
           ('flux_x', boolean), ('flux_t', boolean), ('current_x', boolean),
           ('score_flux_x', type_score_flux),
           ('score_flux_t', type_score_flux),
           ('score_current_x', type_score_current)])
class Tally:
    def __init__(self):
        # ===========
        # Score flags
        # ===========

        # Tracklength
        self.tracklength = False
        self.flux        = False
        self.current     = False
        self.eddington   = False

        # Mesh crossing
        self.crossing    = False
        self.crossing_x  = False
        self.crossing_t  = False
        self.flux_x      = False
        self.flux_t      = False
        self.current_x   = False

        # Uninitialized
        #self.mesh
        #self.score_flux     
        #self.score_flux_x
        #self.score_flux_t
        #self.score_current  
        #self.score_current_x
        #self.score_eddington

    def allocate_bins(self, Ng, N_iter):
        Nt = self.mesh.Nt
        Nx = self.mesh.Nx
        Ny = self.mesh.Ny
        Nz = self.mesh.Nz

        # Tracklength
        if self.flux:
            shape = (N_iter, Ng, Nt, Nx, Ny, Nz)
            self.score_flux = ScoreFlux(shape)
        if self.current:
            shape = (N_iter, Ng, Nt, Nx, Ny, Nz, 3)
            self.score_current = ScoreCurrent(shape)
        if self.eddington:
            shape = (N_iter, Ng, Nt, Nx, Ny, Nz, 6)
            self.score_eddington = ScoreEddington(shape)

        # Mesh crossing
        if self.flux_x:
            shape = (N_iter, Ng, Nt, Nx+1, Ny, Nz)
            self.score_flux_x = ScoreFlux(shape)
        if self.flux_t:
            shape = (N_iter, Ng, Nt+1, Nx, Ny, Nz)
            self.score_flux_t = ScoreFlux(shape)
        if self.current_x:
            shape = (N_iter, Ng, Nt, Nx+1, Ny, Nz, 3)
            self.score_current_x = ScoreCurrent(shape)

        # Set flags
        if self.flux or self.current or self.eddington:
            self.tracklength = True
        if self.flux_x or self.current_x:
            self.crossing = True
            self.crossing_x = True
        if self.flux_t:
            self.crossing = True
            self.crossing_t = True

    def score_tracklength(self, P, d_move):
        # Get indices
        g = P['group']
        t, x, y, z = self.mesh.get_index(P)

        # Score
        flux = d_move*P['weight']
        if self.flux:
            self.score_flux.accumulate(g,t,x,y,z,flux,P)
        if self.current:
            self.score_current.accumulate(g,t,x,y,z,flux,P)
        if self.eddington:
            self.score_eddington.accumulate(g,t,x,y,z,flux,P)

    def score_crossing_x(self, P, t, x, y, z):
        # Get indices
        g = P['group']
        if P['direction']['x'] > 0.0:
            x += 1

        # Score
        flux = P['weight']/abs(P['direction']['x'])
        if self.flux_x:
            self.score_flux_x.accumulate(g,t,x,y,z,flux,P)
        if self.current_x:
            self.score_current_x.accumulate(g,t,x,y,z,flux,P)

    def score_crossing_t(self, P, t, x, y, z):
        # Get indices
        g  = P['group']
        t += 1

        # Score
        flux = P['weight']*P['speed']
        if self.flux_t:
            self.score_flux_t.accumulate(g,t,x,y,z,flux,P)

    def closeout_history(self):
        # Tracklength
        if self.flux:
            closeout_history(self.score_flux)
        if self.current:
            closeout_history(self.score_current)
        if self.eddington:
            closeout_history(self.score_eddington)

        # Mesh crossing
        if self.flux_x:
            closeout_history(self.score_flux_x)
        if self.flux_t:
            closeout_history(self.score_flux_t)
        if self.current_x:
            closeout_history(self.score_current_x)
    
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
