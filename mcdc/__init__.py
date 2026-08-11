"""Public Python interface for MC/DC."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from mcdc.object_.cell import Cell
from mcdc.object_.material import Material
from mcdc.object_.mesh import MeshStructured, MeshUniform
from mcdc.object_.simulation import Simulation
from mcdc.object_.source import Source
from mcdc.object_.surface import Surface
from mcdc.object_.tally import Tally
from mcdc.object_.transport_model_data import NeutronMultigroupData
from mcdc.object_.universe import Lattice, Universe

__all__ = [
    "__version__",
    "Cell",
    "Lattice",
    "Material",
    "MeshStructured",
    "MeshUniform",
    "NeutronMultigroupData",
    "Simulation",
    "Source",
    "Surface",
    "Tally",
    "Universe",
]

try:
    __version__: str = _version("mcdc")
except _PackageNotFoundError:
    __version__ = "unknown"
