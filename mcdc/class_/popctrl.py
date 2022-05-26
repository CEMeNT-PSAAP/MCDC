from   abc   import ABC, abstractmethod
from   math  import floor, ceil
import numba as     nb
import numpy as     np

import mcdc.mpi     as mpi
import mcdc.kernel  as kernel

# Get mcdc global variables as "mcdc"
import mcdc.global_ as mcdc_
mcdc = mcdc_.global_

# ==============================================================================
# PCT: Population Control Technique
# ==============================================================================

class PCT(ABC):
    """Abstract class for Population Control Technique (PCT)"""

    @abstractmethod
    def __call__(self, bank_census, M, bank_soure):
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

class PCT_None(PCT):
    def __init__(self):
        self.type_ = 'None'

    def prepare(self, M):
        return

    def __call__(self, bank_census, M, bank_soure):
        """No PCT, or  analog, only bank-passing"""

        # Accordingly pass/distribute sampled particles
        return mpi.bank_passing(bank)


# ==============================================================================
# Simple Sampling
# ==============================================================================

class PCT_SS(PCT):
    def __init__(self):
        self.type_ = 'SS'

    def prepare(self, M):
        self.count = np.zeros(int(M/mpi.size)*10, dtype=int)

    def __call__(self, bank_census, M, bank_soure):
        """Simple Sampling"""

        # Scan the bank
        idx_start, N_local, N = mpi.bank_scanning(bank)

        # Locally count sampled particles
        for i in range(M):
            xi  = mcdc.rng.random()
            idx = floor(xi*N) - idx_start

            # Local?
            if 0 <= idx and idx < N_local:
                self.count[idx] += 1

        # Set bank_sample
        bank_sample = nb.typed.List.empty_list(type_particle)
        w_factor    = N/M
        for i in range(N_local):
            for j in range(self.count[i]):
                P = bank[i].copy()
                P.weight *= w_factor
                bank_sample.append(P)

            # Reset counter
            self.count[i] = 0

        # Skip ahead RNG
        mcdc.rng.skip_ahead(M)
 
        # Accordingly pass/distribute sampled particles
        return mpi.bank_passing(bank_sample)


# =============================================================================
# Splitting-Roulette
# =============================================================================

class PCT_SR(PCT):
    def __init__(self):
        self.type_ = 'SR'

    def prepare(self, M):
        return

    def __call__(self, bank_census, M, bank_soure):
        """Splitting-Roulette"""

        # Scan the bank
        idx_start, N_local, N = mpi.bank_scanning(bank)

        # Set RNG wrt bank index
        mcdc.rng.skip_ahead(idx_start)

        # Sampling probability
        p = float(M)/float(N)
    
        # Number of splittings
        n_split = floor(p)
    
        # Roulette surviving probability
        p_survive = p - n_split

        # Perform split-roulette to all particles in local bank,
        # and put surviving particles in bank_sample
        bank_sample = nb.typed.List.empty_list(type_particle)
        for P in bank:
            # New weight
            w_prime = P.weight/p

            # Splitting
            for i in range(n_split):
                bank_sample.append(P.copy())
                bank_sample[-1].weight = w_prime

            # Russian roulette
            xi = mcdc.rng.random()
            if xi < p_survive:
                bank_sample.append(P.copy())
                bank_sample[-1].weight = w_prime

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.rng.skip_ahead(N-idx_start)

        # Accordingly pass/distribute sampled particles
        return mpi.bank_passing(bank_sample)


# =============================================================================
# Combing
# =============================================================================

class PCT_CO(PCT):
    def __init__(self):
        self.type_ = 'CO'

    def prepare(self, M):
        return

    def __call__(self, bank_census, M, bank_source):
        """Particle Combing Technique"""

        # Scan the bank
        idx_start, N_local, N = mpi.bank_scanning(bank_census)
        idx_end = idx_start + N_local

        # Teeth distance
        td = N/M

        # Tooth offset
        xi     = mcdc.rng.random()
        offset = xi*td

        # First hiting tooth
        tooth_start = ceil((idx_start-offset)/td)

        # Last hiting tooth
        tooth_end = floor((idx_end-offset)/td) + 1

        # Locally sample particles from census bank
        bank_source['size'] = 0
        for i in range(tooth_start, tooth_end):
            tooth = i*td+offset
            idx   = floor(tooth) - idx_start
            P = bank_census['particles'][idx].copy()
            # Set weight
            P['weight'] *= td
            kernel.add_particle(P, bank_source)

        # Accordingly pass/distribute sampled particles
        #return mpi.bank_passing(bank_sample)


# =============================================================================
# New Combing
# =============================================================================

class PCT_COX(PCT):
    def __init__(self):
        self.type_ = 'COX'

    def prepare(self, M):
        return

    def __call__(self, bank_census, M, bank_soure):
        """New Particle Combing Technique"""

        # Scan the bank
        idx_start, N_local, N = mpi.bank_scanning(bank)
        idx_end = idx_start + N_local

        # Teeth distance
        td = N/M

        # First possible hiting tooth index (and set rng base)
        tooth_start = floor(idx_start/td)
        mcdc.rng.skip_ahead(tooth_start)
        mcdc.rng.rebase()

        # Last possible hiting tooth
        tooth_end = ceil(idx_end/td)

        # Locally sample particles from bank
        bank_sample = nb.typed.List.empty_list(type_particle)
        for i in range(tooth_start, tooth_end):
            # Tooth
            xi    = mcdc.rng.random()
            tooth = (xi+i)*td

            # Check if local
            if tooth >= idx_start and tooth < idx_end:
                idx   = floor(tooth) - idx_start
                P = bank[idx].copy()
                # Set weight
                P.weight *= td
                bank_sample.append(P)

        # Skip ahead RNG (skipping the numbers used for popctrl)
        mcdc.rng.skip_ahead(M-tooth_start)
        
        # Accordingly pass/distribute sampled particles
        return mpi.bank_passing(bank_sample)


# ==============================================================================
# Duplicate-Discard
# ==============================================================================

class PCT_DD(PCT):
    def __init__(self):
        self.type_ = 'DD'

    def prepare(self, M):
        self.count = np.zeros(int(M/mpi.size)*10, dtype=int)
        self.discard_flag = np.full((M*10,1), False)
        return

    def __call__(self, bank_census, M, bank_soure):
        """Duplicate-Discard"""

        # Scan the bank
        idx_start, N_local, N = mpi.bank_scanning(bank)

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
                xi  = mcdc.rng.random()
                idx = floor(xi*N) - idx_start

                # Local?
                if 0 <= idx and idx < N_local:
                    self.count[idx] += 1

            # Set bank_sample
            bank_sample = nb.typed.List.empty_list(type_particle)
            w_factor    = N/M
            for i in range(N_local):
                for j in range(self.count[i]):
                    P = bank[i].copy()
                    P.weight *= w_factor
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
                    xi = mcdc.rng.random()
                    idx = floor(xi*N)

                    # Flag site if not discarded yet
                    if not self.discard_flag[idx]:
                        self.discard_flag[idx] = True
                        break
        
                    # If the site is already discarded, we resample index.
                    # In other words, we are performing a rejection sampling.
    
            # Copy the un-discarded sites
            bank_sample = nb.typed.List.empty_list(type_particle)
            w_factor    = N/M
            for i in range(N_local):
                idx = idx_start + i
                if not self.discard_flag[idx]:
                    P = bank[i].copy()
                    P.weight *= w_factor
                    bank_sample.append(P)
            
            # Reset flag
            for i in range(N): self.discard_flag[i] = False

        # Skip ahead RNG
        mcdc.rng.skip_ahead(N_sample)
 
        # Accordingly pass/distribute sampled particles
        return mpi.bank_passing(bank_sample)
