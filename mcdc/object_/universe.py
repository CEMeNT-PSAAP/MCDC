from __future__ import annotations
from types import NoneType
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from mcdc.object_.cell import Cell

####

import numpy as np

from numpy import int64
from numpy._typing import NDArray

####

from mcdc.constant import INF
from mcdc.object_.base import MCDCObject
from mcdc.util import flatten

# ======================================================================================
# Universe
# ======================================================================================


class Universe(MCDCObject):
    """Reusable collections of cells in the simulation geometry.

    Parameters
    ----------
    name : str, optional
        User-facing universe name.
    cells : list of Cell, optional
        Cells belonging to the universe. Cells are tested in list order by the
        geometry search.

    Examples
    --------
    Group two cells into a reusable universe:

    >>> import numpy as np
    >>> import mcdc
    >>> left = mcdc.Surface.PlaneX(x=-1.0)
    >>> middle = mcdc.Surface.PlaneX(x=0.0)
    >>> right = mcdc.Surface.PlaneX(x=1.0)
    >>> material = mcdc.Material.multigroup(capture=np.array([1.0]))
    >>> cells = [
    ...     mcdc.Cell(region=+left & -middle, fill=material),
    ...     mcdc.Cell(region=+middle & -right, fill=material),
    ... ]
    >>> universe = mcdc.Universe(name="Two regions", cells=cells)

    Create a universe for a spherical inclusion and its surrounding material:

    >>> sphere = mcdc.Surface.Sphere(radius=0.5)
    >>> fuel = mcdc.Material.multigroup(
    ...     fission=np.array([0.2]), nu_p=np.array([2.5])
    ... )
    >>> water = mcdc.Material.multigroup(capture=np.array([0.01]))
    >>> pin = mcdc.Universe(
    ...     name="Pin",
    ...     cells=[
    ...         mcdc.Cell(region=-sphere, fill=fuel),
    ...         mcdc.Cell(region=+sphere, fill=water),
    ...     ],
    ... )

    Use a universe as a cell fill:

    >>> placed_pin = mcdc.Cell(fill=pin, translation=[1.0, 0.0, 0.0])
    """

    # MC/DC framework metadata
    label = "universe"

    name: str
    cells: list[Cell]

    def __init__(self, name: str = "", cells: list[Cell] = []) -> None:
        super().__init__()

        self.name = name or "(Unnamed universe)"
        self.cells = cells

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        text += f"  - Cells: {', '.join(x.name for x in self.cells)}\n"
        return text


# ======================================================================================
# Lattice
# ======================================================================================


class Lattice(MCDCObject):
    """Repeated arrangements of universes in the simulation geometry.

    Parameters
    ----------
    name : str, optional
        User-facing lattice name.
    x, y, z : tuple of (float, float, int), optional
        ``(origin, spacing, number_of_bins)`` for each finite lattice axis, in
        cm. An omitted axis is treated as a single unbounded bin.
    universes : nested list of Universe, optional
        Universe layout supplied in ``[z][y][x]`` order. The y and z axes are
        reversed internally to match MC/DC's Cartesian indexing convention.

    Notes
    -----
    A lattice retains the supplied :class:`Universe` objects. When the owning
    simulation is compiled, those universes are registered and the packed
    lattice IDs are rebuilt from their simulation-local IDs.

    Examples
    --------
    Place two universes next to each other along x:

    >>> import mcdc
    >>> left = mcdc.Universe(name="Left")
    >>> right = mcdc.Universe(name="Right")
    >>> lattice = mcdc.Lattice(
    ...     x=(-1.0, 1.0, 2),
    ...     universes=[[left, right]],
    ... )

    Build a two-dimensional 2-by-2 lattice:

    >>> u00 = mcdc.Universe(name="Lower left")
    >>> u10 = mcdc.Universe(name="Lower right")
    >>> u01 = mcdc.Universe(name="Upper left")
    >>> u11 = mcdc.Universe(name="Upper right")
    >>> lattice_xy = mcdc.Lattice(
    ...     x=(-1.0, 1.0, 2),
    ...     y=(-1.0, 1.0, 2),
    ...     universes=[
    ...         [u00, u10],
    ...         [u01, u11],
    ...     ],
    ... )

    Place the lattice inside a cell:

    >>> lattice_cell = mcdc.Cell(fill=lattice_xy)
    """

    # MC/DC framework metadata
    label = "lattice"
    non_numba = ["universes"]

    name: str

    x0: float
    dx: float
    Nx: int

    y0: float
    dy: float
    Ny: int

    z0: float
    dz: float
    Nz: int

    universes: list[Universe]  # Non-numba
    universe_IDs: Annotated[NDArray[int64], ("Nx", "Ny", "Nz")]

    def __init__(
        self,
        name: str = "",
        x: tuple[float, float, int] | NoneType = None,
        y: tuple[float, float, int] | NoneType = None,
        z: tuple[float, float, int] | NoneType = None,
        universes: list[Universe] = [],
    ) -> None:
        super().__init__()

        self.name = name or "(Unnamed lattice)"
        self.universes = universes

        # Default uniform grids
        self.x0 = -INF
        self.dx = 2 * INF
        self.Nx = 1
        self.y0 = -INF
        self.dy = 2 * INF
        self.Ny = 1
        self.z0 = -INF
        self.dz = 2 * INF
        self.Nz = 1
        self.t0 = 0.0  # Placeholder time grid is needed to use mesh indexing function
        self.dt = INF
        self.Nt = 1

        # Set the grid
        if x is not None:
            self.x0 = x[0]
            self.dx = x[1]
            self.Nx = x[2]
        if y is not None:
            self.y0 = y[0]
            self.dy = y[1]
            self.Ny = y[2]
        if z is not None:
            self.z0 = z[0]
            self.dz = z[1]
            self.Nz = z[2]

        self._set_universe_IDs()

    def _compile_into_simulation(self, simulation) -> bool:
        """Compile contained universes and rebuild their lattice IDs."""
        if not super()._compile_into_simulation(simulation):
            return False

        for universe in flatten(self.universes):
            universe._compile_into_simulation(simulation)

        self._set_universe_IDs()
        return True

    def _set_universe_IDs(self) -> None:
        """Build the packed universe-ID array from the universe layout."""
        # Set universe IDs
        get_ID = np.vectorize(lambda obj: obj.ID)
        universe_IDs = get_ID(self.universes)
        ax_expand = []
        if self.dx == 2 * INF:
            ax_expand.append(2)
        if self.dy == 2 * INF:
            ax_expand.append(1)
        if self.dz == 2 * INF:
            ax_expand.append(0)
        for ax in ax_expand:
            universe_IDs = np.expand_dims(universe_IDs, axis=ax)

        # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
        universe_IDs = np.transpose(universe_IDs)
        universe_IDs = np.flip(universe_IDs, axis=1)
        universe_IDs = np.flip(universe_IDs, axis=2)
        self.universe_IDs = np.array(universe_IDs)

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        text += f"  - (x0, dx, Nx): ({self.x0}, {self.dx}, {self.Nx})\n"
        text += f"  - (y0, dy, Ny): ({self.y0}, {self.dy}, {self.Ny})\n"
        text += f"  - (z0, dz, Nz): ({self.z0}, {self.dz}, {self.Nz})\n"
        text += f"Universes: {set([x.name for x in list(flatten(self.universes))])}"
        return text
