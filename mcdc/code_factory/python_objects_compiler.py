from mcdc.object_.base import MCDCObject
from mcdc.object_.simulation import Simulation


def compile_simulation(simulationPy: Simulation):
    from mcdc.object_.base import MCDC_OBJECT_LABELS
    from mcdc.object_.data import DataNone
    from mcdc.object_.distribution import DistributionNone

    # Reset derived object lists
    simulationPy._reset_object_lists()

    # Object IDs counter
    # TODO: Remove
    next_ID = dict.fromkeys(MCDC_OBJECT_LABELS, 0)

    # Compile reserved objects
    compile_object(simulationPy.root_universe, simulationPy)

    # Root universe
    simulationPy.universes.append(simulationPy.root_universe)

    # "None" objects
    simulationPy.data.append(DataNone())
    simulationPy.distributions.append(DistributionNone())

    # Assign IDs
    assign_IDs(simulationPy.data[0], next_ID, compile_ID)
    assign_IDs(simulationPy.distributions[0], next_ID, compile_ID)
    assign_IDs(simulationPy.universes[0], next_ID, compile_ID)

    # ==================================================================================
    # Compile model
    # ==================================================================================

    root_universe = simulationPy.universes[0]

    # Loop over root universe cells
    for cell in root_universe.cells:
        assign_IDs(cell, next_ID, compile_ID)

        # Fill
        fill = cell.fill

        ## Material
        if isinstance(fill, MaterialBase):
            simulationPy.materials.append(fill)
            assign_IDs(fill, next_ID, compile_ID)

        ## Universe
        elif isinstance(fill, Universe):
            simulationPy.universes.append(fill)
            assign_IDs(fill, next_ID, compile_ID)

        ## Lattice
        elif isinstance(fill, Lattice):
            simulationPy.lattices.append(fill)
            assign_IDs(fill, next_ID, compile_ID)


def compile_object(object_: MCDCObject, simulationPy: Simulation) -> None:
    object_type = type(object_)
    print(object_type)
    matching_members = {
        k: v
        for k, v in vars(simulationPy).items()
        if isinstance(v, list) and all(isinstance(x, object_type) for x in v)
    }
    print(matching_members)
    exit()

    # Assign ID

    object_.ID = next_ID[object_.label]
    object_.compile_ID = compile_ID
    next_ID[object_.label] += 1
