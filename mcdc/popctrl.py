import numpy as np

from abc    import ABC, abstractmethod
from mpi4py import MPI
from copy   import deepcopy
from math   import floor

import mcdc.random
import mcdc.mpi

from mcdc.mpi import bank_passing


class PopCtrl(ABC):
    @abstractmethod
    def __call__(self, bank, N, normalize=False):
        """
        Return a population-controlled bank based on the given bank 
        and the population target N.

        Nearest neighbor bank-passing algorithm [Romano, NSE 2012] is used for
        more efficient parallel communications.

        Normalization (to the value N) of total particle weight is needed 
        for eigenvalue problems.
        """
        pass

class PopCtrlSimple(PopCtrl):
    def __call__(self, bank, N, normalize=False):
        """
        Uniformly sample (in serial) N surviving particles from global bank. 

        EXACTLY yield N surviving particles. 
        Only applicable for uniform weight population.

        Normalization is irrelevant with the use of this method.
        """
        
        # Starting and ending indices in the global bank, and its total size
        N_local = len(bank)
        i_start, i_end, N_global = mcdc.mpi.global_idx(N_local, return_total=True)

        # Locally sample particles from bank
        bank_sample  = []
        sample_count = np.zeros(N_local, dtype=int)
        for i in range(N):
            xi  = mcdc.random.rng()
            idx = floor(xi*N_global)
            # Count if it is local
            if i_start <= idx and idx < i_end:
                idx_local = idx - i_start 
                sample_count[idx_local] += 1
        # Set bank_sample
        for i in range(N_local):
            for j in range(sample_count[i]):
                bank_sample.append(deepcopy(bank[i]))
        
        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(N, change_base=True, stride=1)

        # Starting and ending indices in the global bank_sample
        N_local = len(bank_sample)
        i_start, i_end = mcdc.mpi.global_idx(N_local) 
        
        # Accordingly pass/distribute sampled particles
        return bank_passing(bank_sample, i_start, i_end)


class PopCtrlSR(PopCtrl):
    def __call__(self, bank, N, normalize=False):
        """
        Perform split-roulette (in parallel) to all particles in local banks.
        
        ON AVERAGE yield N surviving particles. 
        
        Normalization is optional.
        """

        # Starting and ending indices in the global bank, and its total size
        N_local = len(bank)
        i_start, i_end, N_global = mcdc.mpi.global_idx(N_local, return_total=True)

        # Set RNG wrt bank index
        mcdc.random.rng.skip_ahead(i_start, stride=1)

        # Total weight of global bank
        w_local = 0.0
        for P in bank: w_local += P.wgt
        w_global = mcdc.mpi.global_wgt(w_local, total_only=True)

        # Surviving particle weight
        w_survive = w_global/N

        # Perform split-roulette to all particles in local bank, 
        #   put surviving particles in bank_sample
        bank_sample  = []
        for P in bank:
            # Surviving probability
            prob_survive = P.wgt/w_survive
            P.wgt        = w_survive

            # Splitting
            n_survive = floor(prob_survive)
            for i in range(n_survive):
                bank_sample.append(deepcopy(P))

            # Russian roulette
            xi            = mcdc.random.rng()
            prob_survive -= n_survive
            if xi < prob_survive:
                bank_sample.append(deepcopy(P))

        # Rebase RNG (skipping the numbers used for popctrl)
        mcdc.random.rng.skip_ahead(N_global, change_base=True, stride=1)

        # Starting and ending indices in the global bank_sample
        N_local = len(bank_sample)
        i_start, i_end, N_global = mcdc.mpi.global_idx(N_local, return_total=True)

        # Redistribute work
        mcdc.mpi.distribute_work(N_global[0])

        # Accordingly pass/distribute sampled particles
        bank_final = bank_passing(bank_sample, i_start, i_end)

        # Normalize?
        if normalize:
            w_normalized = N/N_global
            for P in bank_final:
                P.wgt = w_normalized

        return bank_final


class PopCtrlComb(PopCtrl):
    def __call__(self, bank, N_target, normalize=False):
        #TODO
        """
        Perform particle combing to global bank.
        
        EXACTLY yield N surviving particles. 
        
        Normalization is optional.
        """
        pass
