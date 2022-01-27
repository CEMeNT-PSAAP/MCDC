import numpy as np

from abc    import ABC, abstractmethod
from mpi4py import MPI
from math   import floor, ceil

import mcdc.random
import mcdc.mpi

from mcdc.mpi      import bank_scanning, bank_passing
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
    
    @abstractmethod
    def prepare(self, M):
        """
        Allocate counters and flags, if needed

        ...

        Parameters
        ----------
        M : int
            Population target size
        """

        pass


# ==============================================================================
# NONE
# ==============================================================================

class PCT_NONE(PCT):
    def prepare(self, M):
        return

    def __call__(self, bank, M):
        """No PCT, or  analog, only bank-passing"""

        # Accordingly pass/distribute sampled particles
        return bank_passing(bank)


# ==============================================================================
# Simple Sampling
# ==============================================================================

class PCT_SS(PCT):
    def prepare(self, M):
        self.count = np.zeros(int(M/mcdc.mpi.size)*10, dtype=int)

    def __call__(self, bank, M):
        """Simple Sampling"""

        # Scan the bank
        idx_start, N_local, N = bank_scanning(bank)

        # Locally count sampled particles
        for i in range(M):
            xi  = mcdc.random.rng()
            idx = floor(xi*N) - idx_start

            # Local?
            if 0 <= idx and idx < N_local:
                self.count[idx] += 1

        # Set bank_sample
        bank_sample = []
        w_factor    = N/M
        for i in range(N_local):
            for j in range(self.count[i]):
                P = bank[i].create_copy()
                P.wgt *= w_factor
                bank_sample.append(P)

            # Reset counter
            self.count[i] = 0

        # Skip ahead RNG
        mcdc.random.rng.skip_ahead(M, stride=1, rebase=True)
 
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


# =============================================================================
# Splitting-Roulette
# =============================================================================

class PCT_SR(PCT):
    def prepare(self, M):
        return

    def __call__(self, bank, M):
        """Splitting-Roulette"""

        # Scan the bank
        idx_start, N_local, N = bank_scanning(bank)

        # Set RNG wrt bank index
        mcdc.random.rng.skip_ahead(idx_start, stride=1, rebase=True)

        # Sampling probability
        p = float(M)/float(N)
    
        # Number of splittings
        n_split = floor(p)
    
        # Roulette surviving probability
        p_survive = p - n_split

        # Perform split-roulette to all particles in local bank,
        # and put surviving particles in bank_sample
        bank_sample = []
        for P in bank:
            # New weight
            w_prime = P.wgt/p

            # Splitting
            for i in range(n_split):
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_prime

            # Russian roulette
            xi = mcdc.random.rng()
            if xi < p_survive:
                bank_sample.append(P.create_copy())
                bank_sample[-1].wgt = w_prime

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(N-idx_start, rebase=True, stride=1)

        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


# =============================================================================
# Combing
# =============================================================================

class PCT_CO(PCT):
    def prepare(self, M):
        return

    def __call__(self, bank, M):
        """Particle Combing Technique"""

        # Scan the bank
        idx_start, N_local, N = bank_scanning(bank)
        idx_end = idx_start + N_local

        # Teeth distance
        td = N/M

        # Tooth offset
        xi     = mcdc.random.rng()
        offset = xi*td

        # First hiting tooth
        tooth_start = ceil((idx_start-offset)/td)

        # Last hiting tooth
        tooth_end = floor((idx_end-offset)/td) + 1

        # Locally sample particles from bank
        bank_sample = []
        for i in range(tooth_start, tooth_end):
            tooth = i*td+offset
            idx   = floor(tooth) - idx_start
            P = bank[idx].create_copy()
            # Set weight
            P.wgt *= td
            bank_sample.append(P)

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.rebase()
        
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


# =============================================================================
# New Combing
# =============================================================================

class PCT_COX(PCT):
    def prepare(self, M):
        return

    def __call__(self, bank, M):
        """New Particle Combing Technique"""

        # Scan the bank
        idx_start, N_local, N = bank_scanning(bank)
        idx_end = idx_start + N_local

        # Teeth distance
        td = N/M

        # First possible hiting tooth index (and set rng base)
        tooth_start = floor(idx_start/td)
        mcdc.random.rng.skip_ahead(tooth_start, stride=1, rebase=True)

        # Last possible hiting tooth
        tooth_end = ceil(idx_end/td)

        # Locally sample particles from bank
        bank_sample = []
        for i in range(tooth_start, tooth_end):
            # Tooth
            xi    = mcdc.random.rng()
            tooth = (xi+i)*td

            # Check if local
            if tooth >= idx_start and tooth < idx_end:
                idx   = floor(tooth) - idx_start
                P = bank[idx].create_copy()
                # Set weight
                P.wgt *= td
                bank_sample.append(P)

        # Skip ahead RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(M-tooth_start, stride=1, rebase=True)
        
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)


# ==============================================================================
# Duplicate-Discard
# ==============================================================================

class PCT_DD(PCT):
    def prepare(self, M):
        self.count = np.zeros(int(M/mcdc.mpi.size)*10, dtype=int)
        self.discard_flag = np.full((M*10,1), False)
        return

    def __call__(self, bank, M):
        """Duplicate-Discard"""

        # Scan the bank
        idx_start, N_local, N = bank_scanning(bank)

        # Duplicate or Discard?
        if M >= N:
            # =================================================================
            # Duplicate
            # =================================================================

            # Copies reserved
            N_copy = floor(M/N)

            N_sample = M - N_copy*N

            # Locally count sampled particles
            for i in range(N_local):
                self.count[i] = N_copy

            for i in range(N_sample):
                xi  = mcdc.random.rng()
                idx = floor(xi*N) - idx_start

                # Local?
                if 0 <= idx and idx < N_local:
                    self.count[idx] += 1

            # Set bank_sample
            bank_sample = []
            w_factor    = N/M
            for i in range(N_local):
                for j in range(self.count[i]):
                    P = bank[i].create_copy()
                    P.wgt *= w_factor
                    bank_sample.append(P)

                # Reset counter
                self.count[i] = 0.0
        else:
            # =================================================================
            # Discard
            # =================================================================

            N_sample = N - M

            for i in range(N_sample):
                while True:
                    # Sample discard index
                    xi = mcdc.random.rng()
                    idx = floor(xi*N)

                    # Flag site if not discarded yet
                    if not self.discard_flag[idx]:
                        self.discard_flag[idx] = True
                        break
        
                    # If the site is already discarded, we resample index.
                    # In other words, we are performing a rejection sampling.
    
            # Copy the un-discarded sites
            bank_sample = []
            w_factor    = N/M
            for i in range(N_local):
                idx = idx_start + i
                if not self.discard_flag[idx]:
                    P = bank[i].create_copy()
                    P.wgt *= w_factor
                    bank_sample.append(P)
            
            # Reset flag
            for i in range(N): self.discard_flag[i] = False

        # Skip ahead RNG
        mcdc.random.rng.skip_ahead(N_sample, stride=1, rebase=True)
 
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample)
