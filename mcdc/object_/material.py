from types import NoneType

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from mcdc.object_.base import MCDCObject
from mcdc.object_.element import Element
from mcdc.object_.transport_model import NeutronMultigroup
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.util import ISOTOPIC_ABUNDANCE
from mcdc.print_ import print_error

# ======================================================================================
# Material
# ======================================================================================


class Material(MCDCObject):
    """Define a material from native compositions and/or multigroup data.

    Parameters
    ----------
    name : str, optional
        User-facing material name.
    nuclide_composition : dict of str to float, optional
        Nuclide names and atomic densities in atoms/(barn cm).
    element_composition : dict of str to float, optional
        Element symbols and atomic densities in atoms/(barn cm).
    temperature : float, optional
        Material temperature in kelvin. Each nuclide uses the closest
        temperature available in the data library.
    neutron_multigroup : NeutronMultigroup, optional
        Multigroup cross sections for neutron transport. When supplied with a
        native composition, its ``energy_grid`` is required and defines the
        energy range over which multigroup physics applies.

    Notes
    -----
    At least one native composition or multigroup dataset must be supplied.
    Nuclide and element compositions cannot both be specified because they are
    alternative native representations of the same material. A native
    composition may be supplied together with ``neutron_multigroup`` for hybrid
    neutron physics. Hybrid materials require an explicit ``NeutronMultigroup``
    energy grid so collision physics can select between native and multigroup
    data.

    Nuclide and element objects are created immediately; their data-library
    properties are loaded when the material is compiled into a simulation.
    Construction therefore does not access the data library.

    Examples
    --------
    Define uranium dioxide from nuclide atomic densities:

    >>> import mcdc
    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2, "O16": 4.5e-2},
    ...     temperature=293.6,
    ... )

    Define a one-group multigroup material:

    >>> import numpy as np
    >>> absorber = mcdc.Material(
    ...     name="Absorber",
    ...     neutron_multigroup=mcdc.NeutronMultigroup(capture=np.array([1.0])),
    ... )

    Attach native and multigroup neutron data to the same material:

    >>> hybrid_fuel = mcdc.Material(
    ...     name="Hybrid fuel",
    ...     nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2},
    ...     neutron_multigroup=mcdc.NeutronMultigroup(
    ...         capture=np.array([0.10]),
    ...         fission=np.array([0.20]),
    ...         nu_p=np.array([2.50]),
    ...         energy_grid=np.array([1.0e-5, 20.0e6]),
    ...     ),
    ... )
    """

    # MC/DC framework metadata
    label = "material"
    non_numba = ["nuclide_composition", "element_composition"]

    name: str
    temperature: float
    fissionable: bool
    has_neutron_multigroup: bool

    nuclide_composition: dict[Nuclide, float]  # Non-Numba
    element_composition: dict[Element, float]  # Non-Numba
    neutron_multigroup: NeutronMultigroup

    nuclides: list[Nuclide]
    elements: list[Element]
    nuclide_densities: NDArray[float64]
    element_densities: NDArray[float64]

    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict[str, float] | NoneType = None,
        element_composition: dict[str, float] | NoneType = None,
        temperature: float = 293.6,
        neutron_multigroup: NeutronMultigroup | NoneType = None,
    ) -> None:
        super().__init__()

        # Normalize optional compositions without mutable argument defaults
        nuclide_composition = nuclide_composition or {}
        element_composition = element_composition or {}

        # Require one valid material-data representation
        if nuclide_composition and element_composition:
            print_error(
                "Cannot specify both nuclide_composition and element_composition."
            )
        if (
            not nuclide_composition
            and not element_composition
            and neutron_multigroup is None
        ):
            print_error(
                "Material requires nuclide_composition, element_composition, "
                "or neutron_multigroup."
            )
        if neutron_multigroup is not None and not isinstance(
            neutron_multigroup, NeutronMultigroup
        ):
            print_error("neutron_multigroup must be a NeutronMultigroup object.")
        native_composition_supplied = bool(nuclide_composition or element_composition)
        if (
            native_composition_supplied
            and neutron_multigroup is not None
            and neutron_multigroup.G > 0
            and not neutron_multigroup.has_energy_grid
        ):
            print_error(
                "Material with both a native composition and neutron_multigroup "
                "requires neutron_multigroup.energy_grid."
            )

        # Initialize shared material state
        self.name = name or "(Unnamed material)"
        self.temperature = float(temperature)

        # Use a zero-group placeholder until compilation can select ID 0
        self.neutron_multigroup = (
            neutron_multigroup
            if neutron_multigroup is not None
            else NeutronMultigroup()
        )
        self.has_neutron_multigroup = self.neutron_multigroup.G > 0
        self.fissionable = self.neutron_multigroup.fissionable

        # Create lightweight native-composition objects without loading data
        nearest_temperature = _get_supported_temperature(self.temperature)
        self.nuclide_composition = {
            Nuclide(name, nearest_temperature): float(density)
            for name, density in nuclide_composition.items()
        }
        self.element_composition = {
            Element(name): float(density)
            for name, density in element_composition.items()
        }

        # Initialize the packed native-composition representation
        self.nuclides = list(self.nuclide_composition)
        self.elements = list(self.element_composition)
        self.nuclide_densities = np.asarray(
            list(self.nuclide_composition.values()), dtype=float64
        )
        self.element_densities = np.asarray(
            list(self.element_composition.values()), dtype=float64
        )

    def _compile_into_simulation(self, simulation) -> bool:
        """Canonicalize owned data and register the unified material."""
        # Skip canonicalization when this material is already compiled
        if self.compile_ID == simulation.compile_ID:
            return False

        # Resolve native composition objects before generic member traversal
        nuclide_composition = {}
        element_composition = {}

        for element, density in self.element_composition.items():
            element = _get_or_create_element(element.name, simulation, element)
            element_composition[element] = density

        for nuclide, density in self.nuclide_composition.items():
            nuclide = _get_or_create_nuclide(
                nuclide.name, nuclide.temperature, simulation, nuclide
            )
            nuclide_composition[nuclide] = density

        # Synchronize packed fields with canonical native objects
        self.nuclide_composition = nuclide_composition
        self.element_composition = element_composition
        self.nuclides = list(nuclide_composition)
        self.elements = list(element_composition)
        self.nuclide_densities = np.asarray(
            list(nuclide_composition.values()), dtype=float64
        )
        self.element_densities = np.asarray(
            list(element_composition.values()), dtype=float64
        )

        # Point absent multigroup data to the reserved neutron model
        if self.neutron_multigroup.G == 0:
            self.neutron_multigroup = simulation.neutron_multigroup[0]
        self.has_neutron_multigroup = self.neutron_multigroup.G > 0

        # Register the material, then compile only canonical owned members
        if not super()._compile_into_simulation(simulation):
            return False

        # Resolve fissionability from every available neutron representation
        self.fissionable = self.neutron_multigroup.fissionable or any(
            nuclide.fissionable for nuclide in self.nuclides
        )
        return True

    def __repr__(self) -> str:
        text = super().__repr__()
        text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        text += f"  - Temperature: {self.temperature} K\n"

        if self.nuclide_composition:
            text += "  - Nuclide composition [atoms/barn-cm]\n"
            for nuclide, density in self.nuclide_composition.items():
                text += f"    - {nuclide.name:<5} | {density}\n"

        if self.element_composition:
            text += "  - Element composition [atoms/barn-cm]\n"
            for element, density in self.element_composition.items():
                text += f"    - {element.name:<5} | {density}\n"

        if self.neutron_multigroup.G > 0:
            text += "  - Neutron multigroup model\n"
            text += f"    - G: {self.neutron_multigroup.G}\n"
            text += f"    - J: {self.neutron_multigroup.J}\n"
        return text


# Currently supported temperatures
TEMPERATURES = [0.1, 233.15, 273.15, 293.6, 600.0, 900.0, 1200.0, 2500.0]


# ======================================================================================
# Native-composition helpers
# ======================================================================================


def set_nuclides_from_elements(material, simulation):
    """Expand an elemental composition and register its natural isotopes."""
    material.nuclides = []
    material.nuclide_composition = {}
    nuclide_densities = []

    # Select the nearest supported native-data temperature
    nearest_temperature = _get_supported_temperature(material.temperature)

    # Expand each natural element into its normalized isotopic composition
    for element, element_density in material.element_composition.items():
        norm = sum(ISOTOPIC_ABUNDANCE[element.name].values())

        for nuclide_name, abundance in ISOTOPIC_ABUNDANCE[element.name].items():
            nuclide = _get_or_create_nuclide(
                nuclide_name, nearest_temperature, simulation
            )
            nuclide._compile_into_simulation(simulation)

            nuclide_density = element_density * abundance / norm
            material.nuclides.append(nuclide)
            nuclide_densities.append(nuclide_density)
            material.nuclide_composition[nuclide] = nuclide_density

    material.nuclide_densities = np.asarray(nuclide_densities, dtype=float64)


def set_elements_from_nuclides(material, simulation):
    """Collapse a nuclide composition and register its elements."""
    material.elements = []
    material.element_composition = {}

    # Gather the unique element names represented by the nuclides
    element_names = []
    for nuclide in material.nuclides:
        element_name = nuclide.name[:2]
        if element_name[1].isdigit():
            element_name = element_name[0]
        if element_name not in element_names:
            element_names.append(element_name)
    element_densities = np.zeros(len(element_names), dtype=float64)

    # Accumulate each element's density and register its canonical object
    for i, element_name in enumerate(element_names):
        element = _get_or_create_element(element_name, simulation)
        element._compile_into_simulation(simulation)
        material.elements.append(element)

        density = 0.0
        for nuclide, nuclide_density in material.nuclide_composition.items():
            if nuclide.name[: len(element_name)] != element_name:
                continue
            density += nuclide_density

        element_densities[i] = density
        material.element_composition[element] = density

    material.element_densities = element_densities


def _get_supported_temperature(temperature):
    return min(TEMPERATURES, key=lambda value: abs(value - temperature))


def _get_or_create_element(element_name, simulation, candidate=None):
    for element in simulation.elements:
        if element.name == element_name:
            return element
    return candidate or Element(element_name)


def _get_or_create_nuclide(nuclide_name, temperature, simulation, candidate=None):
    for nuclide in simulation.nuclides:
        if nuclide.name == nuclide_name and nuclide.temperature == temperature:
            return nuclide
    return candidate or Nuclide(nuclide_name, temperature)


def update_fissionable_from_nuclides(material):
    """Update fissionability from native and multigroup neutron data."""
    material.fissionable = material.neutron_multigroup.fissionable or any(
        nuclide.fissionable for nuclide in material.nuclides
    )
