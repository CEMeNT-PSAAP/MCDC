import importlib.metadata

from mcdc.input_ import (
    nuclide,
    material,
    surface,
    cell,
    universe,
    lattice,
    source,
    setting,
    eigenmode,
    implicit_capture,
    weighted_emission,
    population_control,
    branchless_collision,
    time_census,
    weight_window,
    iQMC,
    weight_roulette,
    IC_generator,
    uq,
    reset,
    domain_decomposition,
    make_particle_bank,
    save_particle_bank,
)
import mcdc.tally
from mcdc.main import (
    prepare,
    run,
    visualize,
)

__version__ = importlib.metadata.version("mcdc")
