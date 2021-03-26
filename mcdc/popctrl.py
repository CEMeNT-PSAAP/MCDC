from abc    import ABC, abstractmethod
from mpi4py import MPI

import mcdc.random


class PopCtrl(ABC):
    @abstractmethod
    def __call__(self, bank, N_target, normalize):
        pass

class PopCtrlSimple(PopCtrl):
    def __call__(self, bank, N_target, normalize):
        pass

class PopCtrlSplitRoulette(PopCtrl):
    def __call__(self, bank, N_target, normalize):
        pass

class PopCtrlCombing(PopCtrl):
    def __call__(self, bank, N_target, normalize):
        pass
