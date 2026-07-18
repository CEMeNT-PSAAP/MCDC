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
from mcdc.object_.data import DataBase, DataNone
from mcdc.object_.distribution import DistributionBase, DistributionNone
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
    """MC/DC transport simulation.

    A ``Simulation`` represents a complete transport calculation. It combines
    the physical model, source definitions, simulation settings, tally
    definitions, and transport techniques required to execute a Monte Carlo
    simulation.

    A simulation is first constructed using Python objects. Before execution,
    the model is compiled into an execution-ready representation optimized for
    MC/DC's transport engine.

    Notes
    -----
    The Python object graph is independent of the compiled representation used
    during transport. Compilation traverses the object graph, assigns internal
    identifiers, validates the model, and generates the data structures used by
    the execution backend.
    """

    _next_compile_ID: int = 1  # Non-Numba

    # Basic parameters
    name: str
    compile_ID: int  # Non-Numba
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
        """Create a simulation.

        Parameters
        ----------
        name : str, optional
            User-defined simulation name. If omitted, an unnamed simulation is
            created.
        """
        non_numba = [
            "regions",
            "bank_active",
            "bank_census",
            "bank_source",
            "bank_future",
            "_next_compile_ID",
            "compiled",
            "compile_ID",
        ]
        super().__init__("simulation", non_numba)

        # Set name
        self.name = name or "(Unnamed simulation)"

        self.compile_ID = 0
        self.compiled = False

        # ==============================================================================
        # Simulation objects
        # ==============================================================================

        # Physics
        self.data = [DataNone()]
        self.distributions = [DistributionNone()]
        self.neutron_reactions = []
        self.electron_reactions = []
        self.nuclides = []
        self.elements = []
        self.materials = []
        self.sources = []

        # Geometry
        self.surfaces = []
        self.regions = []
        self.cells = []
        self.universes = [Universe("Root Universe")]
        self.lattices = []
        self.meshes = []

        # Tallies
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

    # ==================================================================================
    # Simulation object setters
    # ==================================================================================

    def set_model(self, cells: Sequence[Cell]) -> None:
        """Set the simulation model.

        Parameters
        ----------
        cells : Sequence[Cell]
            Cells that define the root universe of the model.

        Notes
        -----
        Only the root cells are stored. Connected geometry and physics objects are
        discovered automatically during compilation.
        """
        self.universes[0].cells = list(cells)
        self.compiled = False

    def set_sources(self, sources: Sequence[Source]) -> None:
        """Set the simulation source definitions.

        Parameters
        ----------
        sources : Sequence[Source]
            Source definitions used to initialize particles.

        Notes
        -----
        Connected distributions and supporting objects are collected during
        compilation.
        """
        self.sources = list(sources)
        self.compiled = False

    def set_tallies(self, tallies: Sequence[Tally]) -> None:
        """Set the simulation tally definitions.

        Parameters
        ----------
        tallies : Sequence[Tally]
            Tallies used to score transport quantities.

        Notes
        -----
        Connected meshes, filters, and other referenced objects are collected
        during compilation.
        """
        self.tallies = list(tallies)
        self.compiled = False

    def set_root_universe(self, cells: Sequence[Cell]) -> None:
        """Set the cells of the root universe.

        Parameters
        ----------
        cells : Sequence[Cell]
            Cells contained directly in the root universe.

        Notes
        -----
        The root universe occupies index ``0`` in the compiled universe array.
        Geometry tracking begins from this universe by default.
        """
        self.universes[0].cells = list(cells)

    # ==================================================================================
    # Operations
    # ==================================================================================

    def compile(self) -> None:
        """Compile the simulation.

        Compilation validates the simulation, discovers all connected objects,
        assigns internal identifiers, and generates the execution-ready data
        structures used by the transport engine.

        Notes
        -----
        Compilation does not perform particle transport. It only prepares the
        simulation for execution or other operations that require the compiled
        representation.
        """
        from mcdc.main import compile_simulation

        compile_ID = type(self)._next_compile_ID

        try:
            compile_simulation(self, compile_ID)
        except Exception:
            self.compiled = False
            raise

        self.compile_ID = compile_ID
        type(self)._next_compile_ID += 1
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
        """Visualize the simulation model.

        Generate two-dimensional slices of the simulation geometry at one or more
        times. If necessary, the simulation is compiled automatically before
        visualization.

        Parameters
        ----------
        vis_plane : {'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}
            Coordinate plane to visualize.

        x : float or array_like
            Plane x-coordinate for ``'yz'`` visualization, or x-axis range for
            ``'xy'`` and ``'xz'`` visualizations.

        y : float or array_like
            Plane y-coordinate for ``'xz'`` visualization, or y-axis range for
            ``'xy'`` and ``'yz'`` visualizations.

        z : float or array_like
            Plane z-coordinate for ``'xy'`` visualization, or z-axis range for
            ``'xz'`` and ``'yz'`` visualizations.

        pixels : array_like
            Number of pixels along the two visualization axes.

        colors : array_like
            Sequence of ``(material, color)`` pairs used to render the model.

        time : float or array_like
            Time or times at which geometry snapshots are generated.

        save_as : str, optional
            Output filename. If omitted, the visualization is displayed without
            saving.

        Notes
        -----
        Visualization uses the compiled model representation but does not perform
        particle transport or modify the compiled simulation state.
        """
        if not self.compiled:
            self.compile()

        from mcdc.visualize import visualize_model

        visualize_model(self, vis_plane, x, y, z, pixels, colors, time, save_as)

    def run(self) -> None:
        """Execute the transport simulation.

        If necessary, the simulation is compiled automatically before execution.

        Notes
        -----
        Compilation is invalidated after execution because transport updates the
        internal simulation state. If the Python model is modified directly without
        using the public simulation interface, users are responsible for ensuring
        the simulation is recompiled before subsequent operations.
        """
        from mcdc.main import run_simulation

        if not self.compiled:
            self.compile()

        try:
            run_simulation(self)
        finally:
            self.compiled = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"compiled={self.compiled}, "
            f"compile_ID={self.compile_ID})"
        )
