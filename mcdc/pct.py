import numpy as np

from abc    import ABC, abstractmethod
from mpi4py import MPI
from math   import floor, ceil

import mcdc.random
import mcdc.mpi

from mcdc.mpi      import bank_passing
from mcdc.particle import Particle
from mcdc.misc     import binary_search

import sys


# ==============================================================================
# PCT: Population Control Technique
# ==============================================================================

class PCT(ABC):
    """Abstract class for Population Control Technique (PCT)"""

    @abstractmethod
    def __call__(self, bank, M):
        """
        Return a controlled particle bank based on the given initial
        bank `bank` and the population target `M`

        ...

        Parameters
        ----------
        bank : vector of Particle
            Particle initial population
        M : int
            Population target size
        """

        pass


# ==============================================================================
# Simple Sampling
# ==============================================================================

class PCT_SSU(PCT):
    def __call__(self, bank, M):
        """Simple Sampling - Uniform"""

        # Starting and ending of global indices
        N_local = len(bank)
        i_start, i_end = mcdc.mpi.global_idx(N_local)

        # Broadcast total size
        N = mcdc.mpi.bcast(np.array([i_end], dtype=int), mcdc.mpi.last)

        # Locally count sampled particles
        sample_count = np.zeros(N_local, dtype=int)
        for i in range(M):
            xi  = mcdc.random.rng()
            idx = floor(xi*N)

            # Local?
            if i_start <= idx and idx < i_end:
                idx_local = idx - i_start
                sample_count[idx_local] += 1

        # Set bank_sample
        bank_sample = []
        w_factor    = N/M
        for i in range(N_local):
            for j in range(sample_count[i]):
                P = bank[i].create_copy()
                # Set weight
                P.wgt *= w_factor
                bank_sample.append(P)

        # Skip ahead RNG
        mcdc.random.rng.skip_ahead(M, stride=1, rebase=True)
 
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


class PCT_SSW(PCT):
    def __call__(self, bank, M):
        """Simple Sampling - Weight-based"""

        # Starting and ending indices in the global bank
        N_local = len(bank)
        i_start, i_end = mcdc.mpi.global_idx(N_local)

        # Broadcast total size
        N = mcdc.mpi.bcast(np.array([i_end], dtype=int), mcdc.mpi.last)

        # Weight CDF
        w_cdf = np.zeros(N+1)
        for i in range(N_local):
            idx = i_start + i
            w_cdf[idx+1] = w_cdf[idx] + bank[i].wgt
        for i in range(N - i_end):
            idx = i_end + i
            w_cdf[idx+1] = w_cdf[idx]
        # Parallel reduce
        buff = np.zeros_like(w_cdf)
        mcdc.mpi.allreduce(w_cdf, buff)
        w_cdf = buff

        # Total weight
        W = w_cdf[-1]

        # Locally count sampled particles
        sample_count = np.zeros(N_local, dtype=int)
        for i in range(M):
            xi  = mcdc.random.rng()
            wgt = xi*W
            idx = binary_search(wgt,w_cdf)
            
            # Local?
            if i_start <= idx and idx < i_end:
                idx_local = idx - i_start
                sample_count[idx_local] += 1

        # Set bank_sample
        bank_sample  = []
        w_prime = W/M
        for i in range(N_local):
            for j in range(sample_count[i]):
                P = bank[i].create_copy()
                # Set weight
                P.wgt = w_prime
                bank_sample.append(P)
        
        # Skip ahead RNG
        mcdc.random.rng.skip_ahead(M, stride=1, rebase=True)
 
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


# =============================================================================
# Splitting-Roulette
# =============================================================================

class PCT_SRU(PCT):
    def __call__(self, bank, M):
        """Splitting-Roulette - Uniform"""

        # Starting and ending indices in the global bank
        N_local = len(bank)
        i_start, i_end = mcdc.mpi.global_idx(N_local)

        # Broadcast total size
        N = mcdc.mpi.bcast(np.array([i_end], dtype=int), mcdc.mpi.last)

        # Set RNG wrt bank index
        mcdc.random.rng.skip_ahead(i_start, stride=1, rebase=True)

        # Surviving probability
        ps = float(M)/float(N)

        # Perform split-roulette to all particles in local bank, 
        # and put surviving particles in bank_sample
        bank_sample  = []
        idx = 0
        for P in bank:
            # Surviving probability and weight
            prob_survive = ps
            w_survive    = P.wgt/prob_survive

            # Splitting
            n_survive = floor(prob_survive)
            for i in range(n_survive):
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_survive

            # Russian roulette
            xi            = mcdc.random.rng()
            prob_survive -= n_survive
            if xi < prob_survive:
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_survive
            idx +=1

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(N-i_start, rebase=True, stride=1)

        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample, redistribute=True)


class PCT_SRW(PCT):
    def __call__(self, bank, M):
        """Splitting-Roulette - Weight-based"""

        # Starting and ending indices in the global bank
        N_local = len(bank)
        i_start, i_end = mcdc.mpi.global_idx(N_local)

        # Broadcast total size
        N = mcdc.mpi.bcast(np.array([i_end], dtype=int), mcdc.mpi.last)

        # Set RNG wrt bank index
        mcdc.random.rng.skip_ahead(i_start, stride=1, rebase=True)

        # Total weight of global bank
        W_local = np.zeros(1)
        for P in bank:
            W_local[0] += P.wgt
        buff = np.zeros(1)
        mcdc.mpi.allreduce(W_local, buff)
        W = buff[0]

        # Surviving weight
        w_survive = W/M

        # Perform split-roulette to all particles in local bank, 
        # and put surviving particles in bank_sample
        bank_sample  = []
        idx = 0
        for P in bank:
            # Surviving probability
            prob_survive = P.wgt/w_survive

            # Splitting
            n_survive = floor(prob_survive)
            for i in range(n_survive):
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_survive

            # Russian roulette
            xi            = mcdc.random.rng()
            prob_survive -= n_survive
            if xi < prob_survive:
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_survive
            idx +=1

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(N-i_start, rebase=True, stride=1)

        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample, redistribute=True)


# =============================================================================
# Combing
# =============================================================================

class PCT_COU(PCT):
    def __call__(self, bank, M):
        """Particle Combing Technique - Uniform"""

        # Starting and ending indices in the global bank
        N_local = len(bank)
        i_start, i_end = mcdc.mpi.global_idx(N_local)

        # Broadcast total size
        N = mcdc.mpi.bcast(np.array([i_end], dtype=int), mcdc.mpi.last)

        # Teeth distance
        td = N/M

        # Tooth offset
        xi     = mcdc.random.rng()
        offset = xi*td

        # First hiting tooth
        tooth_start = ceil((i_start-offset)/td)

        # Last hiting tooth
        tooth_end = floor((i_end-offset)/td) + 1

        # Locally sample particles from bank
        bank_sample = []
        idx = 0
        for i in range(tooth_start, tooth_end):
            tooth = i*td+offset
            idx   = floor(tooth) - i_start
            P = bank[idx].create_copy()
            # Set weight
            P.wgt *= td
            bank_sample.append(P)

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.rebase()
        
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


class PCT_COW(PCT):
    def __call__(self, bank, M):
        """Particle Combing Technique - Weight-based [Booth 1996]"""

        # Weight CDF
        N_local = len(bank)
        w_cdf   = np.zeros(N_local+1)
        for i in range(N_local):
            w_cdf[i+1] = w_cdf[i] + bank[i].wgt
        
        # Starting and ending weight CDF in the global bank
        w_start, w_end = mcdc.mpi.global_wgt(w_cdf[-1])

        # Readjust local weight CDF
        w_cdf += w_start

        # Broadcast total weight
        W = mcdc.mpi.bcast(np.array([w_end]), mcdc.mpi.last)

        # Teeth distance
        w_prime = W/M

        # Tooth offset
        xi       = mcdc.random.rng()
        w_offset = xi*w_prime

        # First hiting tooth
        tooth_start = ceil((w_cdf[0]-w_offset)/w_prime)

        # Last hiting tooth
        tooth_end = floor((w_cdf[-1]-w_offset)/w_prime) + 1

        # Locally sample particles from bank
        bank_sample = []
        idx = 0
        for i in range(tooth_start, tooth_end):
            wgt = (xi+i)*w_prime
            idx += binary_search(wgt,w_cdf[idx:])
            P = bank[idx].create_copy()
            # Set weight
            P.wgt = w_prime
            bank_sample.append(P)

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.rebase()
        
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)
