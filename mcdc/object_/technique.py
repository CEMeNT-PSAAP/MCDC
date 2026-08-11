import numpy as np

from mcdc.constant import INF
from mcdc.object_.base import MCDCBase
from mcdc.object_.mesh import MeshBase, MeshUniform
from mcdc.print_ import print_error
from numpy.typing import NDArray
from typing import Annotated

# ======================================================================================
# Neutron multigroup
# ======================================================================================


class NeutronMultigroup(MCDCBase):
    """Describe whether neutron multigroup transport is standard or hybrid."""

    # MC/DC framework metadata
    label = "neutron_multigroup"

    hybrid: bool  # Whether neutron multigroup transport is hybrid

    def __init__(self) -> None:
        self.hybrid = True


# ======================================================================================
# Implicit capture
# ======================================================================================


class ImplicitCapture(MCDCBase):
    """Simulation-owned implicit-capture configuration."""

    # MC/DC framework metadata
    label = "implicit_capture"

    active: bool

    def __init__(self) -> None:
        self.active = False

    def __call__(self, active: bool = True) -> None:
        """Configure implicit capture.

        Parameters
        ----------
        active : bool, optional
            Whether implicit capture is enabled.

        Examples
        --------
        Enable implicit capture:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.implicit_capture()

        Disable implicit capture:

        >>> simulation.technique.implicit_capture(active=False)
        """
        self.active = active


# ======================================================================================
# Weighted emission
# ======================================================================================


class WeightedEmission(MCDCBase):
    """Simulation-owned weighted-emission configuration."""

    # MC/DC framework metadata
    label = "weighted_emission"

    active: bool
    weight_target: float

    def __init__(self) -> None:
        self.active = False
        self.weight_target = 0.0

    def __call__(self, active: bool = True, weight_target: float = 1.0) -> None:
        """Configure weighted emission.

        Parameters
        ----------
        active : bool, optional
            Whether the technique is active.
        weight_target : float, optional
            Target statistical weight for emitted particles.

        Examples
        --------
        Enable weighted emission with unit target weight:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.weighted_emission(weight_target=1.0)

        Select a different target weight:

        >>> simulation.technique.weighted_emission(weight_target=0.5)

        Disable weighted emission:

        >>> simulation.technique.weighted_emission(active=False)
        """
        self.active = active
        self.weight_target = weight_target


# ======================================================================================
# Weight roulette
# ======================================================================================


class GlobalWeightRoulette(MCDCBase):
    """Simulation-owned global weight-roulette configuration."""

    # MC/DC framework metadata
    label = "global_weight_roulette"

    active: bool
    weight_threshold: float
    weight_target: float

    def __init__(self) -> None:
        self.active = False
        self.weight_threshold = 0.0
        self.weight_target = 1.0

    def __call__(
        self, weight_threshold: float = 0.0, weight_target: float = 1.0
    ) -> None:
        """Enable roulette below a global weight threshold.

        Parameters
        ----------
        weight_threshold : float, optional
            Particle weight below which roulette is applied.
        weight_target : float, optional
            Statistical weight assigned to particles that survive roulette.
            Must be greater than or equal to ``weight_threshold``.

        Examples
        --------
        Apply roulette below a particle weight of 0.25 and raise surviving
        particles to unit weight:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.global_weight_roulette(
        ...     weight_threshold=0.25,
        ...     weight_target=1.0,
        ... )

        Use a lower target weight:

        >>> simulation.technique.global_weight_roulette(
        ...     weight_threshold=0.1,
        ...     weight_target=0.5,
        ... )
        """
        if weight_threshold > weight_target:
            print_error(
                "For weight roulette, weight threshold has to be smaller than the target"
            )
        self.active = True
        self.weight_threshold = weight_threshold
        self.weight_target = weight_target


# ======================================================================================
# Weight Windows
# ======================================================================================


class WeightWindows(MCDCBase):
    """Simulation-owned particle weight-window configuration."""

    # MC/DC framework metadata
    label = "weight_windows"

    active: bool

    # energy
    energy_bounds: NDArray[np.float64]
    Ne: int
    # space
    mesh: MeshBase
    Nx: int
    Ny: int
    Nz: int

    # arrays of ww params
    lower_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]
    target_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]
    upper_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]

    def __init__(self) -> None:
        self.active = False
        self.energy_bounds = np.array([0.0, 1.0])
        self.Ne = 1
        self.mesh_ID = -1  # skirt around having to create a MeshBase instance
        self.Nx, self.Ny, self.Nz = 1, 1, 1
        self.Nt = 1
        shape = (self.Ne, self.Nx, self.Ny, self.Nz)
        self.lower_weights = np.array([1.0]).reshape(*shape)
        self.target_weights = np.array([1.0]).reshape(*shape)
        self.upper_weights = np.array([1.0]).reshape(*shape)

    def __call__(self, weight_windows, mesh=None, energy=None) -> None:
        """Configure lower, target, and upper particle weights.

        Parameters
        ----------
        weight_windows : ndarray, shape (Ne, Nx, Ny, Nz, 3)
            Lower, target, and upper weights in the final dimension. Every
            lower weight must be positive, and each window must satisfy
            ``lower <= target <= upper``.
        mesh : MeshUniform or MeshStructured, optional
            Spatial mesh. The default is one unbounded uniform bin.
        energy : ndarray, optional
            Strictly increasing energy boundaries in eV. The default is one
            all-energy bin.

        Examples
        --------
        Apply one weight window over all space and energy:

        >>> import numpy as np
        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> windows = np.array([0.5, 1.0, 2.0]).reshape(1, 1, 1, 1, 3)
        >>> simulation.technique.weight_windows(windows)

        Configure weight windows on a uniform spatial mesh:

        >>> mesh = mcdc.MeshUniform(x=(-5.0, 1.0, 10))
        >>> windows = np.tile([0.25, 0.5, 1.0], (1, 10, 1, 1, 1))
        >>> simulation.technique.weight_windows(windows, mesh=mesh)

        Configure both energy- and space-dependent windows:

        >>> energy = np.array([0.0, 0.625, 20.0e6])
        >>> windows = np.tile([0.25, 0.5, 1.0], (2, 10, 1, 1, 1))
        >>> simulation.technique.weight_windows(
        ...     windows,
        ...     mesh=mesh,
        ...     energy=energy,
        ... )
        """
        # fill in defaults
        if mesh is None:
            mesh = MeshUniform()
        if energy is None:
            # usable for both groups and max energy
            energy = np.array([-0.5, INF])

        # get mesh size
        match mesh.label:
            case "uniform_mesh":
                nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
            case "structured_mesh":
                nx, ny, nz = (
                    mesh.x.shape[0] - 1,
                    mesh.y.shape[0] - 1,
                    mesh.z.shape[0] - 1,
                )
            case _:
                print_error(
                    f"{type(mesh).__name__} is not supported for weight windows"
                )
        # validate energy as strictly increasing
        if not (np.diff(energy) > 0).all():
            print_error("Energy bounds must be strictly increasing")
        # get energy size
        if len(energy.shape) != 1:
            print_error(
                f"Invalid shape for energy; expected 1D got {len(energy.shape)}D"
            )
        ne = energy.shape[0] - 1

        # check correct shape
        mesh_shape = (nx, ny, nz)
        ww_shape = weight_windows.shape
        expected_shape = (ne, *mesh_shape, 3)
        if ww_shape != expected_shape:
            print_error(
                f"Weight window array has shape {ww_shape}, but expected {expected_shape}"
            )

        self.active = True
        self.energy_bounds = energy
        self.Ne = ne
        self.mesh = mesh
        self.Nx, self.Ny, self.Nz = mesh_shape
        shape = (self.Ne, *mesh_shape)
        self.lower_weights = weight_windows[..., 0]
        self.target_weights = weight_windows[..., 1]
        self.upper_weights = weight_windows[..., 2]

        # check weight windows are valid
        if (self.lower_weights <= 0.0).any():
            print_error(
                "Lower bound weights must be strictly positive to avoid invalid roulette behavior"
            )
        if (self.lower_weights > self.target_weights).any():
            print_error(
                "Lower bound weight can not be greater than the target weight for any weight window"
            )
        if (self.target_weights > self.upper_weights).any():
            print_error(
                "Target weight can not be greater than the upper bound weight for any weight window"
            )


# ======================================================================================
# Population control
# ======================================================================================


class PopulationControl(MCDCBase):
    """Simulation-owned source-bank population-control configuration."""

    # MC/DC framework metadata
    label = "population_control"

    active: bool

    def __init__(self) -> None:
        self.active = False

    def __call__(self, active: bool = True) -> None:
        """Configure source-bank population control.

        Parameters
        ----------
        active : bool, optional
            Whether source-bank population control is enabled.

        Examples
        --------
        Enable source-bank population control:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.population_control()

        Disable source-bank population control:

        >>> simulation.technique.population_control(active=False)
        """
        self.active = active


# ======================================================================================
# Simulation technique collection
# ======================================================================================


class Technique(MCDCBase):
    """Own all simulation-wide transport-technique configurations.

    Access the individual callable configurations through
    ``simulation.technique``. The same hierarchy is retained in the packed
    runtime simulation.
    """

    # MC/DC framework metadata
    label = "technique"

    neutron_multigroup: NeutronMultigroup
    implicit_capture: ImplicitCapture
    weighted_emission: WeightedEmission
    global_weight_roulette: GlobalWeightRoulette
    weight_windows: WeightWindows
    population_control: PopulationControl

    def __init__(self) -> None:
        # Construct every simulation-wide technique configuration
        self.neutron_multigroup = NeutronMultigroup()
        self.implicit_capture = ImplicitCapture()
        self.weighted_emission = WeightedEmission()
        self.global_weight_roulette = GlobalWeightRoulette()
        self.weight_windows = WeightWindows()
        self.population_control = PopulationControl()
