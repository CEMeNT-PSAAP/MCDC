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
    root_universe: Universe
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
        # MC/DC framework metadata
        super().__init__(
            label="simulation",
            non_numba=[
                "regions",
                "root_universe",
                "bank_active",
                "bank_census",
                "bank_source",
                "bank_future",
                "_next_compile_ID",
                "compiled",
                "compile_ID",
            ],
        )

        self.compile_ID = 0
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
        self.root_universe.cells = list(cells)
        self.compiled = False

    def set_sources(self, sources: Sequence[Source]) -> None:
        self.sources = list(sources)
        self.compiled = False

    def set_tallies(self, tallies: Sequence[Tally]) -> None:
        self.tallies = list(tallies)
        self.compiled = False

    # ==================================================================================
    # Operations
    # ==================================================================================

    def compile(self) -> None:
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
        if not self.compiled:
            self.compile()

        from mcdc.visualize import visualize_model

        visualize_model(self, vis_plane, x, y, z, pixels, colors, time, save_as)

    def run(self) -> None:
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
