import numpy as np

from abc import ABC, abstractmethod

import mcdc.mpi

from mcdc.misc     import binary_search
from mcdc.constant import EPSILON, INF
from mcdc.print    import print_error


class Tally:
    def __init__(self, name, scores, x=None, y=None, z=None, t=None):
        self.name = name

        # Set score
        for s in scores:
            if s not in ['flux', 'current', 'eddington']:
                print_error("Unknown tally score %s"%s)
        self.score_name = scores

        # Cartesian space
        if x is None:
            self.x = np.array([-INF,INF])
        else:
            self.x = x
        if y is None:
            self.y = np.array([-INF,INF])
        else:
            self.y = y
        if z is None:
            self.z = np.array([-INF,INF])
        else:
            self.z = z
        if t is None:
            self.t = np.array([-INF,INF])
        else:
            self.t = t

    def allocate_bins(self, N_iter, Ng):
        Nt = len(self.t)-1
        Nx = len(self.x)-1
        Ny = len(self.y)-1
        Nz = len(self.z)-1

        shape = [N_iter, Nt, Ng, Nx, Ny, Nz]
        self.scores = []
        for s in self.score_name:
            if s == 'flux':
                self.scores.append(ScoreFlux(shape))
            elif s == 'current':
                self.scores.append(ScoreCurrent(shape))
            elif s == 'eddington':
                self.scores.append(ScoreEddington(shape))

    def distance_search(self, value, direction, grid):
        if direction == 0.0:
            return INF
        idx = binary_search(value, grid)
        if direction > 0.0:
            idx += 1
        dist = (grid[idx] - value)/direction
        return dist

    def distance(self, P):
        x = P.position.x
        y = P.position.y
        z = P.position.z
        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        t = P.time
        v = P.speed

        d = INF
        d = min(d, self.distance_search(x, ux, self.x))
        d = min(d, self.distance_search(y, uy, self.y))
        d = min(d, self.distance_search(z, uz, self.z))
        d = min(d, self.distance_search(t, 1.0, self.t))

        return d

    def score(self, P, d_move):
        # Get indices
        t = binary_search(P.time, self.t)
        x = binary_search(P.position.x, self.x)
        y = binary_search(P.position.y, self.y)
        z = binary_search(P.position.z, self.z)

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
            mcdc.mpi.reduce_master(S.sum, S.sum_buff)
            mcdc.mpi.reduce_master(S.sum_sq, S.sum_sq_buff)
            S.sum[:]    = S.sum_buff[:]
            S.sum_sq[:] = S.sum_sq_buff[:]
            
            # Store results
            if mcdc.mpi.master:
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
        if mcdc.mpi.master:
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
