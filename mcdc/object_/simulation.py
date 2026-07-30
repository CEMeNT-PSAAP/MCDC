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

import numpy as np

from collections.abc import Sequence
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
    resulting object graph, assigns IDs, and prepares it for conversion to the
    packed arrays consumed by :mod:`mcdc.transport`.

    Examples
    --------
    Assemble a minimal one-group slab simulation:

    >>> import numpy as np
    >>> import mcdc
    >>> material = mcdc.MaterialMG(capture=np.array([1.0]))
    >>> left = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
    >>> right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    >>> cell = mcdc.Cell(region=+left & -right, fill=material)
    >>> source = mcdc.Source(x=[0.0, 1.0], isotropic=True, energy_group=0)
    >>> tally = mcdc.Tally(cell=cell, scores=["flux"])
    >>> simulation = mcdc.Simulation(name="Slab")
    >>> simulation.set_model([cell])
    >>> simulation.set_sources([source])
    >>> simulation.set_tallies([tally])
    >>> simulation.settings.N_particle = 1_000

    Visualize an x-z slice of the model:

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

    Run particle transport and write the configured output:

    >>> simulation.run()

    Configure a time-dependent calculation with census times:

    >>> transient = mcdc.Simulation(name="Transient slab")
    >>> transient.set_model([cell])
    >>> transient.set_sources([source])
    >>> transient.set_tallies([tally])
    >>> transient.settings.N_particle = 10_000
    >>> transient.settings.set_time_census(
    ...     time=[1.0e-6, 2.0e-6, 5.0e-6],
    ...     tally_frequency=10,
    ... )

    Configure a k-eigenvalue calculation:

    >>> fuel = mcdc.MaterialMG(
    ...     capture=np.array([0.10]),
    ...     fission=np.array([0.20]),
    ...     nu_p=np.array([2.50]),
    ... )
    >>> fuel_cell = mcdc.Cell(region=+left & -right, fill=fuel)
    >>> eigenvalue = mcdc.Simulation(name="Critical slab")
    >>> eigenvalue.set_model([fuel_cell])
    >>> eigenvalue.set_sources([source])
    >>> eigenvalue.settings.N_particle = 10_000
    >>> eigenvalue.settings.set_eigenmode(
    ...     N_inactive=20,
    ...     N_active=100,
    ...     k_init=1.0,
    ... )

    Enable common variance-reduction techniques:

    >>> simulation.implicit_capture()
    >>> simulation.global_weight_roulette(
    ...     weight_threshold=0.25,
    ...     weight_target=1.0,
    ... )
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

    # ==================================================================================
    # Simulation object setters
    # ==================================================================================

    def set_model(self, cells: Sequence[Cell]) -> None:
        """Set the cells in the root universe and invalidate compiled state."""
        self.root_universe.cells = list(cells)
        self.compiled = False

    def set_sources(self, sources: Sequence[Source]) -> None:
        """Set particle sources and invalidate compiled state."""
        self.sources = list(sources)
        self.compiled = False

    def set_tallies(self, tallies: Sequence[Tally]) -> None:
        """Set requested tallies and invalidate compiled state."""
        self.tallies = list(tallies)
        self.compiled = False

    # ==================================================================================
    # Operations
    # ==================================================================================

    def compile(self) -> None:
        """Compile the Python object graph into a new simulation snapshot.

        A globally unique ``compile_ID`` identifies the snapshot. Every
        embedded or registered :class:`~mcdc.object_.base.MCDCBase` reached
        during compilation records that ID.
        """
        from mcdc.code_factory.python_objects_compiler import compile_simulation

        self.compile_ID = type(self)._next_compile_ID
        type(self)._next_compile_ID += 1

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
        """
        if not self.compiled:
            self.compile()

        from mcdc.visualize import visualize_model

        visualize_model(self, vis_plane, x, y, z, pixels, colors, time, save_as)

    def run(self) -> None:
        """Compile if needed, execute particle transport, and write output."""
        from mcdc.main import run_simulation

        if not self.compiled:
            self.compile()

        run_simulation(self)
        self.compiled = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"compiled={self.compiled}, "
            f"compile_ID={self.compile_ID})"
        )
