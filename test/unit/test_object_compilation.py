import numpy as np

import mcdc

from mcdc.object_.base import MCDCBase, MCDCObject
from mcdc.object_.data import DataPolynomial
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.universe import Universe


class ObjectOwner(Universe):
    child: DataPolynomial
    children: list[DataPolynomial]
    ignored: DataPolynomial

    non_numba = ["ignored"]

    def __init__(self, child, children, ignored):
        super().__init__()
        self.child = child
        self.children = children
        self.ignored = ignored


class EmbeddedConfiguration(MCDCBase):
    label = "embedded_configuration"

    def __init__(self):
        self.member = None


def test_mcdc_object_compiles_object_members_and_lists():
    child = DataPolynomial(np.array([1.0]))
    children = [
        DataPolynomial(np.array([2.0])),
        DataPolynomial(np.array([3.0])),
    ]
    ignored = DataPolynomial(np.array([4.0]))
    simulation = mcdc.Simulation()
    simulation.root_universe = ObjectOwner(child, children, ignored)

    simulation.compile()

    assert simulation.data[1:] == [child, *children]
    assert ignored.compile_ID == 0


def test_simulation_compiles_objects_owned_by_embedded_configuration():
    mesh = mcdc.MeshUniform()
    weight_windows = np.ones((1, 1, 1, 1, 3))
    simulation = mcdc.Simulation()
    simulation.weight_windows(weight_windows, mesh=mesh)

    simulation.compile()

    assert simulation.meshes == [mesh]
    assert simulation.weight_windows.compile_ID == simulation.compile_ID


def test_embedded_compile_id_prevents_cycles_and_supports_recompilation():
    configuration_a = EmbeddedConfiguration()
    configuration_b = EmbeddedConfiguration()
    configuration_a.member = configuration_b
    configuration_b.member = configuration_a
    simulation = mcdc.Simulation()
    simulation.configuration = configuration_a

    simulation.compile()
    first_compile_ID = simulation.compile_ID

    assert configuration_a.compile_ID == first_compile_ID
    assert configuration_b.compile_ID == first_compile_ID

    simulation.compile()

    assert simulation.compile_ID != first_compile_ID
    assert configuration_a.compile_ID == simulation.compile_ID
    assert configuration_b.compile_ID == simulation.compile_ID


def test_material_canonicalizes_composition_before_member_compilation(monkeypatch):
    # Replace library loading with the minimal state needed for this unit test
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = nuclide.name == "U235"
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)

    material_a = mcdc.Material(nuclide_composition={"U235": 1.0})
    material_b = mcdc.Material(nuclide_composition={"U235": 2.0})
    simulation = mcdc.Simulation()
    simulation.compile()

    material_a._compile_into_simulation(simulation)
    material_b._compile_into_simulation(simulation)

    assert len(simulation.nuclides) == 1
    assert material_a.nuclides == simulation.nuclides
    assert material_b.nuclides == simulation.nuclides
    assert material_a.fissionable
    assert material_b.fissionable


def test_simulation_compilation_finalizes_model_wide_state():
    source_a = mcdc.Source(probability=1.0)
    source_b = mcdc.Source(probability=3.0)
    simulation = mcdc.Simulation()
    simulation.set_sources([source_a, source_b])
    simulation.settings.neutron_eigenvalue_mode = True
    simulation.settings.N_inactive = 2
    simulation.settings.N_cycle = 4
    simulation.settings.k_init = 1.25

    simulation.compile()

    assert np.allclose([source_a.probability, source_b.probability], [0.25, 0.75])
    assert simulation.settings.neutron_multigroup_mode
    assert simulation.k_eff == 1.25
    assert not simulation.cycle_active
    assert simulation.k_cycle.shape == (4,)
    assert simulation.gyration_radius.shape == (4,)


def test_simulation_compilation_sets_particle_bank_capacities():
    simulation = mcdc.Simulation()
    simulation.settings.N_particle = 100
    simulation.settings.N_census = 2
    simulation.settings.active_bank_buffer = 11
    simulation.settings.census_bank_buffer_ratio = 2.0
    simulation.settings.source_bank_buffer_ratio = 3.0
    simulation.settings.future_bank_buffer_ratio = 4.0

    simulation.compile()

    N_work = int(np.ceil(100 / simulation.mpi_size))
    assert simulation.bank_active.size[0] == 11
    assert simulation.bank_census.size[0] == 2 * N_work
    assert simulation.bank_source.size[0] == 3 * N_work
    assert simulation.bank_future.size[0] == 4 * N_work
