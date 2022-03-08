from   abc   import ABC, abstractmethod
import numpy as     np

from   mcdc.class_.mesh import Mesh
import mcdc.mpi         as mpi
from   mcdc.print_ import print_error

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

class Tally:
    def __init__(self, name, scores, x=None, y=None, z=None, t=None):
        self.name = name

        # Set score
        for s in scores:
            if s not in ['flux', 'current', 'eddington']:
                print_error("Unknown tally score %s"%s)
        self.score_name = scores

        # Set mesh
        self.mesh = Mesh(x,y,z,t)

    def allocate_bins(self):
        Nt = self.mesh.Nt
        Nx = self.mesh.Nx
        Ny = self.mesh.Ny
        Nz = self.mesh.Nz
        Ng = mcdc.cells[0].material.G
        N_iter = mcdc.settings.N_iter

        shape = [N_iter, Nt, Ng, Nx, Ny, Nz]
        self.scores = []
        for s in self.score_name:
            if s == 'flux':
                self.scores.append(ScoreFlux(shape))
            elif s == 'current':
                self.scores.append(ScoreCurrent(shape))
            elif s == 'eddington':
                self.scores.append(ScoreEddington(shape))

    def score(self, P, d_move):
        # Get indices
        t, x, y, z = self.mesh.get_index(P)

        # Get current index
        for S in self.scores:
            S(t, x, y, z, d_move, P)
   
    def closeout_history(self):
        for S in self.scores:
            # Accumulate sums of history
            S.sum    += S.bin
            S.sum_sq += np.square(S.bin)
        
            # Reset bin
            S.bin.fill(0.0)
    
    def closeout(self, N_hist, i_iter):
        for S in self.scores:
            # MPI Reduce
            mpi.reduce_master(S.sum, S.sum_buff)
            mpi.reduce_master(S.sum_sq, S.sum_sq_buff)
            S.sum[:]    = S.sum_buff[:]
            S.sum_sq[:] = S.sum_sq_buff[:]
            
            # Store results
            if mpi.master:
                S.mean[i_iter,:] = S.sum/N_hist
                S.sdev[i_iter,:] = np.sqrt((S.sum_sq/N_hist 
                                            - np.square(S.mean[i_iter]))\
                                           /(N_hist-1))
            
            # Reset history sums
            S.sum.fill(0.0)
            S.sum_sq.fill(0.0)
        

class Score(ABC):
    def __init__(self, shape, name):
        self.name = name

        # History accumulator
        self.bin = np.zeros(shape[1:]) # Skip the N_iter

        # Sums of history
        self.sum    = np.zeros_like(self.bin)
        self.sum_sq = np.zeros_like(self.bin)
        
        # MPI buffers
        self.sum_buff    = np.zeros_like(self.bin)
        self.sum_sq_buff = np.zeros_like(self.bin)
        
        # Results
        if mpi.master:
            self.mean = np.zeros(shape)
            self.sdev = np.zeros_like(self.mean)

    @abstractmethod
    def __call__(self, t, x, y, z, distance, P):
        pass

class ScoreFlux(Score):
    def __init__(self, shape):
        Score.__init__(self, shape, 'flux')
    def __call__(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        self.bin[t, P.group, x, y, z] += flux

class ScoreCurrent(Score):
    def __init__(self, shape):
        Score.__init__(self, shape+[3], 'current')
    def __call__(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        self.bin[t, P.group, x, y, z, 0] += flux*P.direction.x
        self.bin[t, P.group, x, y, z, 1] += flux*P.direction.y
        self.bin[t, P.group, x, y, z, 2] += flux*P.direction.z

class ScoreEddington(Score):
    def __init__(self, shape):
        Score.__init__(self, shape+[6], 'eddington')
    def __call__(self, t, x, y, z, distance, P):
        flux = distance*P.weight
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        self.bin[t, P.group, x, y, z, 0] += flux*ux*ux
        self.bin[t, P.group, x, y, z, 1] += flux*ux*uy
        self.bin[t, P.group, x, y, z, 2] += flux*ux*uz
        self.bin[t, P.group, x, y, z, 3] += flux*uy*uy
        self.bin[t, P.group, x, y, z, 4] += flux*uy*uz
        self.bin[t, P.group, x, y, z, 5] += flux*uz*uz
