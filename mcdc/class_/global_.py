from numba              import types, int64, boolean, float64
from numba.experimental import jitclass
from numba.typed        import List

from mcdc.class_.cell     import type_cell
from mcdc.class_.material import type_material
from mcdc.class_.particle import Bank, type_bank
from mcdc.class_.random   import Random, type_random
from mcdc.class_.setting  import Setting, type_setting
from mcdc.class_.surface  import type_surface
from mcdc.class_.source   import Source, type_source
from mcdc.class_.tally    import Tally, TallyGlobal, type_tally,\
                                 type_tally_global
from mcdc.class_.wwindow  import WeightWindow, type_weight_window

@jitclass([('materials', types.ListType(type_material)),
           ('surfaces', types.ListType(type_surface)),
           ('cells', types.ListType(type_cell)),
           ('source', type_source),
           ('setting', type_setting),
           ('rng', type_random),
           ('tally', type_tally),
           ('tally_global', type_tally_global),
           ('bank', type_bank),
           ('weight_window', type_weight_window),
           ('runtime_total', float64),
           ('runtime_pct', float64),
           ('i_iter', int64),
           ('N_work', int64),
           ('master', boolean)])
class Global:
    def __init__(self):
        self.reset()

    def reset(self):
        # Model
        self.materials = List.empty_list(type_material)
        self.surfaces  = List.empty_list(type_surface)
        self.cells     = List.empty_list(type_cell)
        self.source    = Source()

        # Setting
        self.setting = Setting()

        # Random number generator
        self.rng = Random()

        # Tallies
        self.tally        = Tally()
        self.tally_global = TallyGlobal()

        # Particle banks
        self.bank = Bank()

        # Variance reduction techniques
        self.weight_window = WeightWindow()

        # Runtime
        self.runtime_total = 0.0
        self.runtime_pct   = 0.0

        # Indices
        self.i_iter = 0

        # MPI-related
        self.N_work = 0
        self.master = False
