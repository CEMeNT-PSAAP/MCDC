from   numba              import int64, float64, boolean, types, njit, config
from   numba.experimental import jitclass
import numpy              as     np

from mcdc.class_.mesh  import Mesh, type_mesh

@jitclass([('mesh', type_mesh), ('window', float64[:,:,:,:])])
class WeightWindow:
    def __init__(self):
        pass
        # Uninitialized
        #self.mesh
        #self.window

    def apply_(self, P, mcdc):
        # Get indices
        t, x, y, z = self.mesh.get_index(P)

        # Target weight
        w_target = self.window[t,x,y,z]
       
        # Surviving probability
        p = P.weight/w_target

        # Set target weight
        P.weight = w_target

        # If above target
        if p > 1.0:
            # Splitting (keep the original particle)
            n_split = np.floor(p)
            for i in range(n_split-1):
                mcdc.bank.history.append(P.copy())

            # Russian roulette
            p -= n_split
            xi = mcdc.rng.random()
            if xi <= p:
                mcdc.bank.history.append(P.copy())

        # Below target
        else:
            # Russian roulette
            xi = mcdc.rng.random()
            if xi > p:
                P.alive = False

if not config.DISABLE_JIT:
    type_weight_window = WeightWindow.class_type.instance_type
else:
    type_weight_window = None
