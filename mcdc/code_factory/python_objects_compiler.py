import numpy as np

from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.cell import Region, Cell
from mcdc.object_.data import DataBase, DataNone
from mcdc.object_.distribution import DistributionBase, DistributionNone
from mcdc.object_.electron_reaction import ElectronReactionBase
from mcdc.object_.element import Element
from mcdc.object_.material import MaterialBase
from mcdc.object_.mesh import MeshBase
from mcdc.object_.neutron_reaction import NeutronReactionBase
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.universe import Universe, Lattice
from mcdc.object_.simulation import Simulation
from mcdc.object_.source import Source
from mcdc.object_.surface import Surface
from mcdc.object_.tally import Tally
from mcdc.print_ import print_error

NONE_OBJECT_CLASSES = (DataNone, DistributionNone)  # Has customized compilation


def compile_simulation(simulation: Simulation):
    # Reset model
    simulation._reset_model()

    # Reserved ojects
    none_data = DataNone()
    none_distribution = DistributionNone()

    # Compile reserved objects
    none_data._compile_into_simulation(simulation)
    none_distribution._compile_into_simulation(simulation)

    # Compile model
    root_universe = simulation.root_universe
    root_universe._compile_into_simulation(simulation)

    # Compile source
    sources = simulation.sources
    simulation.sources = []
    for source in sources:
        source._compile_into_simulation(simulation)

    # Compile tally
    tallies = simulation.tallies
    simulation.tallies = []
    for tally in tallies:
        tally._compile_into_simulation(simulation)

    # Apply settings as needed
    settings = simulation.settings
    if simulation.settings.neutron_eigenvalue_mode:
        simulation.k_cycle = np.zeros(settings.N_cycle)
        simulation.gyration_radius = np.zeros(settings.N_cycle)


def register_object(object_: MCDCObject, simulation: Simulation) -> bool:
    # Skip if already compiled
    if object_.compile_ID == simulation.compile_ID:
        return False

    # Get the object list
    if isinstance(object_, Cell):
        object_list = simulation.cells
    elif isinstance(object_, DataBase):
        object_list = simulation.data
    elif isinstance(object_, DistributionBase):
        object_list = simulation.distributions
    elif isinstance(object_, Lattice):
        object_list = simulation.lattices
    elif isinstance(object_, MaterialBase):
        object_list = simulation.materials
    elif isinstance(object_, MeshBase):
        object_list = simulation.meshes
    elif isinstance(object_, Element):
        object_list = simulation.elements
    elif isinstance(object_, ElectronReactionBase):
        object_list = simulation.electron_reactions
    elif isinstance(object_, Nuclide):
        object_list = simulation.nuclides
    elif isinstance(object_, NeutronReactionBase):
        object_list = simulation.neutron_reactions
    elif isinstance(object_, Region):
        object_list = simulation.regions
    elif isinstance(object_, Source):
        object_list = simulation.sources
    elif isinstance(object_, Surface):
        object_list = simulation.surfaces
    elif isinstance(object_, Tally):
        object_list = simulation.tallies
    elif isinstance(object_, Universe):
        object_list = simulation.universes
    else:
        object_list = []
        print_error(f"Unidentified object list for object {object_}")

    # Assign IDs
    object_.ID = len(object_list)
    if isinstance(object_, MCDCPolymorphic):
        object_.sub_ID = sum(
            [
                x.sub_type == object_.sub_type
                for x in object_list
                if isinstance(x, MCDCPolymorphic)
            ]
        )
    object_.compile_ID = simulation.compile_ID

    # Assign name if needed
    if hasattr(object_, "name") and object_.name.startswith("(Unnamed"):
        if isinstance(object_, MCDCPolymorphic):
            ID = object_.sub_ID
        else:
            ID = object_.ID
        object_.name = object_.label + f"_{ID}"

    # Register to simulation (TODO: Resolve IDE error message)
    object_list.append(object_)

    return True
