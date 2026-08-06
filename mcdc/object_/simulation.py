from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from mcdc.object_.cell import Cell, Region
    from mcdc.object_.element import Element
    from mcdc.object_.electron_reaction import ElectronReactionBase
    from mcdc.object_.material import MaterialBase
    from mcdc.object_.nuclide import Nuclide
    from mcdc.object_.neutron_reaction import NeutronReactionBase
    from mcdc.object_.source import Source
    from mcdc.object_.surface import Surface
    from mcdc.object_.tally import Tally

####

import math
from collections.abc import Sequence

import numpy as np
from mpi4py import MPI
from numpy import float64, int64
from numpy.typing import NDArray

####

from mcdc.object_.base import MCDCBase
from mcdc.object_.data import DataBase
from mcdc.object_.distribution import DistributionBase
from mcdc.object_.gpu_tools import GPUMeta
from mcdc.object_.mesh import MeshBase
from mcdc.object_.particle import ParticleBank
from mcdc.object_.settings import Settings
from mcdc.object_.technique import (
    ImplicitCapture,
    PopulationControl,
    GlobalWeightRoulette,
    WeightWindows,
    WeightedEmission,
)

from mcdc.object_.universe import Universe, Lattice

# ======================================================================================
# Simulation
# ======================================================================================


class Simulation(MCDCBase):
    """Own a complete MC/DC model, settings, techniques, and runtime state.

    Parameters
    ----------
    name : str, optional
        User-facing simulation name.

    Notes
    -----
    Geometry, sources, and tallies are supplied with :meth:`set_model`,
    :meth:`set_sources`, and :meth:`set_tallies`. :meth:`compile` walks the
    resulting object graph, assigns IDs, and finalizes object-local and
    model-wide state before conversion to the packed arrays consumed by
    :mod:`mcdc.transport`.

    MC/DC uses one active simulation context per Python process. A model-object
    instance belongs to one simulation, although it may be referenced multiple
    times within that model. Construct independent object graphs and use
    separate processes for concurrent simulations.

    Each simulation owns its execution settings. Access them through
    ``simulation.settings`` by assigning values such as
    :attr:`settings.N_particle <mcdc.Simulation.settings.N_particle>` or by
    calling configuration methods such as
    :meth:`settings.set_eigenmode <mcdc.Simulation.settings.set_eigenmode>`.

    Transport techniques are configured directly on the simulation through
    methods such as :meth:`implicit_capture <mcdc.Simulation.implicit_capture>`
    and :meth:`weight_windows <mcdc.Simulation.weight_windows>`.

    Examples
    --------
    Configure commonly adjusted settings:

    >>> import mcdc
    >>> simulation = mcdc.Simulation(name="Slab")
    >>> simulation.settings.N_particle = 10_000
    >>> simulation.settings.N_batch = 20
    >>> simulation.settings.rng_seed = 12345
    >>> simulation.settings.output_name = "slab"
    """

    # MC/DC framework metadata
    label = "simulation"
    non_numba = [
        "_next_compile_ID",
        "compiled",
        "regions",
        "root_universe",
        "bank_active",
        "bank_census",
        "bank_source",
        "bank_future",
    ]
    _next_compile_ID: int = 1  # Non-Numba

    # Basic parameters
    name: str
    compiled: bool  # Non-Numba

    # Physics
    data: list[DataBase]
    distributions: list[DistributionBase]
    neutron_reactions: list[NeutronReactionBase]
    electron_reactions: list[ElectronReactionBase]
    nuclides: list[Nuclide]
    elements: list[Element]
    materials: list[MaterialBase]
    sources: list[Source]

    # Geometry
    surfaces: list[Surface]
    regions: list[Region]  # Non-Numba
    cells: list[Cell]
    universes: list[Universe]
    root_universe: Universe  # Non-Numba
    lattices: list[Lattice]
    meshes: list[MeshBase]

    # Tallies
    tallies: list[Tally]

    # Settings
    settings: Settings

    # Techniques
    implicit_capture: ImplicitCapture
    weighted_emission: WeightedEmission
    global_weight_roulette: GlobalWeightRoulette
    weight_windows: WeightWindows
    population_control: PopulationControl

    # Particle banks
    bank_active: ParticleBank  # Non-Numba
    bank_census: ParticleBank  # Non-Numba
    bank_source: ParticleBank  # Non-Numba
    bank_future: ParticleBank  # Non-Numba

    # Simulation indices
    idx_work: int
    idx_cycle: int
    idx_census: int
    idx_batch: int

    # k-eigenvalue globals
    k_eff: float
    k_cycle: NDArray[float64]
    k_avg: float
    k_sdv: float
    k_avg_running: float
    k_sdv_running: float
    #
    n_avg: float
    n_sdv: float
    n_max: float
    #
    C_avg: float
    C_sdv: float
    C_max: float
    #
    eigenvalue_tally_nuSigmaF: Annotated[NDArray[float64], (1,)]
    eigenvalue_tally_n: Annotated[NDArray[float64], (1,)]
    eigenvalue_tally_C: Annotated[NDArray[float64], (1,)]
    #
    gyration_radius: NDArray[float64]
    #
    cycle_active: bool

    # MPI parameters
    mpi_size: int
    mpi_rank: int
    mpi_master: bool
    mpi_work_start: int
    mpi_work_size: int
    mpi_work_size_total: int
    mpi_work_iter: Annotated[NDArray[int64], (1,)]

    # Runtimes
    runtime_total: float
    runtime_preparation: float
    runtime_simulation: float
    runtime_output: float
    runtime_bank_management: float

    # GPU metadata
    gpu_meta: GPUMeta
    source_seed: int

    def __init__(self, name: str = "") -> None:
        self.compiled = False

        self.name = name or "(Unnamed simulation)"
        self.root_universe = Universe("Root Universe")

        # Initialize with empty model objects
        self._reset_model()

        # Also empty sources and tallies
        self.sources = []
        self.tallies = []

        # ==============================================================================
        # Simulation settings and techniques
        # ==============================================================================

        self.settings = Settings()

        # Techniques
        self.implicit_capture = ImplicitCapture()
        self.weighted_emission = WeightedEmission()
        self.global_weight_roulette = GlobalWeightRoulette()
        self.weight_windows = WeightWindows()
        self.population_control = PopulationControl()

        # ==============================================================================
        # Particle banks
        # ==============================================================================

        self.bank_active = ParticleBank(tag="active")
        self.bank_census = ParticleBank(tag="census")
        self.bank_source = ParticleBank(tag="source")
        self.bank_future = ParticleBank(tag="future")

        # ==============================================================================
        # Simulation variables and parameters
        # ==============================================================================

        # Simulation indices
        self.idx_work = 0
        self.idx_cycle = 0
        self.idx_census = 0
        self.idx_batch = 0

        # Eigenvalue simulation
        self.k_eff = 0.0
        self.k_cycle = np.ones(1)
        self.k_avg = 0.0
        self.k_sdv = 0.0
        self.k_avg_running = 0.0
        self.k_sdv_running = 0.0
        #
        self.n_avg = 0.0  # Neutron density
        self.n_sdv = 0.0
        self.n_max = 0.0
        #
        self.C_avg = 0.0  # Precursor density
        self.C_sdv = 0.0
        self.C_max = 0.0
        #
        self.eigenvalue_tally_nuSigmaF = np.zeros(1)
        self.eigenvalue_tally_n = np.zeros(1)
        self.eigenvalue_tally_C = np.zeros(1)
        #
        self.gyration_radius = np.zeros(1)
        #
        self.cycle_active = False

        # MPI parameters
        self.mpi_size = MPI.COMM_WORLD.Get_size()
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_master = self.mpi_rank == 0
        self.mpi_work_start = 0
        self.mpi_work_size = 0
        self.mpi_work_size_total = 0
        self.mpi_work_iter = np.zeros(1, dtype=int64)

        # Runtimes
        self.runtime_total = 0.0
        self.runtime_preparation = 0.0
        self.runtime_simulation = 0.0
        self.runtime_output = 0.0
        self.runtime_bank_management = 0.0

        # GPU metadata
        self.gpu_meta = GPUMeta()
        self.source_seed = 0

    def _reset_model(self) -> None:
        # Physics
        self.data = []
        self.distributions = []
        self.neutron_reactions = []
        self.electron_reactions = []
        self.nuclides = []
        self.elements = []
        self.materials = []

        # Geometry
        self.surfaces = []
        self.regions = []
        self.cells = []
        self.universes = []
        self.lattices = []
        self.meshes = []

    def _finalize_compilation(self) -> None:
        """Finalize model-wide state after object discovery.

        Object-specific compilation hooks register dependencies and derive
        fields owned by one object. This method handles relationships and
        invariants that require the complete simulation, before the model is
        packed into the runtime representation. Any runtime-visible object
        created here must be compiled explicitly because recursive discovery
        has already completed.
        """
        from mcdc.object_.material import (
            Material,
            MaterialMG,
            set_elements_from_nuclides,
            set_nuclides_from_elements,
            update_fissionable_from_nuclides,
        )

        settings = self.settings

        # Limit transport to the latest requested tally boundary
        settings.time_boundary = min(
            [settings.time_boundary] + [tally.time[-1] for tally in self.tallies]
        )

        # Complete native-material compositions for the transported particles
        for material in self.materials:
            if not isinstance(material, Material):
                continue
            if settings.neutron_transport and len(material.nuclides) == 0:
                set_nuclides_from_elements(material, self)
            if settings.electron_transport and len(material.elements) == 0:
                set_elements_from_nuclides(material, self)

        # Load the physics data required by the completed material model
        if settings.neutron_transport:
            for nuclide in self.nuclides:
                nuclide.set_neutron_data(self)
            for material in self.materials:
                if isinstance(material, Material):
                    update_fissionable_from_nuclides(material)

        if settings.electron_transport:
            for element in self.elements:
                element.set_electron_data(self)

        # Determine the neutron physics representation
        if len(self.materials) == 0:
            settings.neutron_multigroup_mode = True
        else:
            settings.neutron_multigroup_mode = isinstance(self.materials[0], MaterialMG)

        # Derive tally shapes that depend on simulation-wide settings
        if settings.use_census_based_tally:
            for tally in self.tallies:
                tally._use_census_based_tally(settings.census_tally_frequency, self)

        # Normalize source-selection probabilities across the complete source set
        source_probability = sum(source.probability for source in self.sources)
        for source in self.sources:
            source.probability /= source_probability

        # Derive particle-bank capacities from settings and the MPI decomposition
        N_work = math.ceil(settings.N_particle / self.mpi_size)
        N_census = settings.N_census

        if settings.neutron_eigenvalue_mode or N_census == 1:
            settings.future_bank_buffer_ratio = 0.0
        if not settings.neutron_eigenvalue_mode and N_census == 1:
            settings.census_bank_buffer_ratio = 0.0
            settings.source_bank_buffer_ratio = 0.0

        self.bank_active.size[0] = settings.active_bank_buffer
        self.bank_census.size[0] = int(settings.census_bank_buffer_ratio * N_work)
        self.bank_source.size[0] = int(settings.source_bank_buffer_ratio * N_work)
        self.bank_future.size[0] = int(settings.future_bank_buffer_ratio * N_work)

        # Initialize run state derived from the compiled settings
        self.k_eff = settings.k_init
        self.cycle_active = (
            not settings.neutron_eigenvalue_mode or settings.N_inactive == 0
        )
        if settings.neutron_eigenvalue_mode:
            self.k_cycle = np.zeros(settings.N_cycle)
            self.gyration_radius = np.zeros(settings.N_cycle)

    # ==================================================================================
    # Simulation object setters
    # ==================================================================================

    def set_model(self, cells: Sequence[Cell]) -> None:
        """Set the root cells that define a complete model (geometry and materials) of
        the simulation.

        Pass only cells that belong directly to the root universe. Do not include
        cells nested inside subuniverses; they are discovered automatically during
        model traversal as long as they are reachable through the root cells.

        Parameters
        ----------
        cells : sequence of Cell
            Cells to place in the root universe.

        Examples
        --------
        Attach previously constructed cells:

        >>> simulation.set_model([fuel_cell, moderator_cell])
        """
        self.root_universe.cells = list(cells)
        self.compiled = False

    def set_sources(self, sources: Sequence[Source]) -> None:
        """Set particle sources for the simulation.

        Parameters
        ----------
        sources : sequence of Source
            Particle sources to sample during transport.

        Examples
        --------
        Attach previously constructed sources:

        >>> simulation.set_sources([volume_source, boundary_source])
        """
        self.sources = list(sources)
        self.compiled = False

    def set_tallies(self, tallies: Sequence[Tally]) -> None:
        """Set requested tallies for the simulation.

        Parameters
        ----------
        tallies : sequence of Tally
            Tallies to score during transport.

        Examples
        --------
        Attach previously constructed tallies:

        >>> simulation.set_tallies([flux_tally, current_tally])
        """
        self.tallies = list(tallies)
        self.compiled = False

    # ==================================================================================
    # Operations
    # ==================================================================================

    def compile(self) -> None:
        """Compile and finalize the Python model into a simulation snapshot.

        A globally unique ``compile_ID`` identifies the snapshot. Command-line overrides
        are applied before any derived state is resolved. Every embedded or registered
        :class:`~mcdc.object_.base.MCDCBase` reached during compilation records that ID.
        Object-local hooks discover and prepare their dependencies, then model-wide
        finalization resolves state that requires the complete simulation.
        """
        from mcdc.config import override_settings
        from mcdc.code_factory.python_objects_compiler import compile_simulation

        self.compile_ID = type(self)._next_compile_ID
        type(self)._next_compile_ID += 1

        override_settings(self)

        compile_simulation(self)
        self.compiled = True

    def visualize_model(
        self,
        vis_plane,
        x,
        y,
        z,
        pixels,
        colors,
        time,
        save_as,
    ) -> None:
        """Render a two-dimensional material map of the compiled model.

        Parameters are forwarded to :func:`mcdc.visualize.visualize_model`.
        The model is compiled first when necessary.

        Examples
        --------
        Render an x-z slice of the model:

        >>> simulation.visualize_model(
        ...     vis_plane="xz",
        ...     x=[0.0, 1.0],
        ...     y=0.0,
        ...     z=[-0.5, 0.5],
        ...     pixels=(100, 100),
        ...     colors=None,
        ...     time=[0.0],
        ...     save_as="slab",
        ... )
        """
        if not self.compiled:
            self.compile()

        from mcdc.visualize import visualize_model

        visualize_model(self, vis_plane, x, y, z, pixels, colors, time, save_as)

    def run(self) -> None:
        """Compile when needed, execute transport, and write output.

        Examples
        --------
        Run a fully configured simulation:

        >>> simulation.run()
        """
        if not self.compiled:
            self.compile()

        from mcdc.main import run_simulation

        run_simulation(self)
        self.compiled = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"compiled={self.compiled}, "
            f"compile_ID={self.compile_ID})"
        )
