import numpy as np

from numpy import float64
from numpy.typing import NDArray
from types import NoneType
from typing import Annotated

####

from mcdc.constant import MATERIAL, MATERIAL_MG
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.element import Element
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.util import ISOTOPIC_ABUNDANCE
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Material base class
# ======================================================================================


class MaterialBase(MCDCPolymorphic):
    """Base class shared by continuous-energy and multigroup materials.

    Parameters
    ----------
    name : str
        User-facing material name.
    """

    # MC/DC framework metadata
    label = "material"
    sub_type = -1  # Polymorphic base

    name: str
    fissionable: bool

    def __init__(self, name: str) -> None:
        super().__init__()

        self.name = name or "(Unnamed material)"
        self.fissionable = False

    def __repr__(self):
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        return text


# ======================================================================================
# Native material
# ======================================================================================


class Material(MaterialBase):
    """Define a continuous-energy material from a data-library composition.

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

    Notes
    -----
    Exactly one of ``nuclide_composition`` or ``element_composition`` must be
    supplied. Nuclide and element objects are created immediately; their
    data-library properties are loaded when the material is compiled into a
    simulation. Construction therefore does not access the data library.
    Before compiling, visualizing, or running the model, ``MCDC_LIB`` must
    identify the directory containing the required HDF5 files.

    Examples
    --------
    Define uranium dioxide from nuclide atomic densities:

    >>> import mcdc
    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2, "O16": 4.5e-2},
    ...     temperature=293.6,
    ... )

    Define a material from natural elemental compositions:

    >>> steel = mcdc.Material(
    ...     name="Steel",
    ...     element_composition={"Fe": 8.0e-2, "C": 8.0e-4},
    ...     temperature=600.0,
    ... )

    Define a single-isotope material at another supported temperature:

    >>> moderator = mcdc.Material(
    ...     name="Hydrogen",
    ...     nuclide_composition={"H1": 6.7e-2},
    ...     temperature=273.15,
    ... )
    """

    # MC/DC framework metadata
    label = "native_material"
    sub_type = MATERIAL
    non_numba = ["nuclide_composition", "element_composition"]

    nuclide_composition: dict[Nuclide, float]  # Non-numba
    element_composition: dict[Element, float]  # Non-numba

    nuclides: list[Nuclide]
    elements: list[Element]
    nuclide_densities: NDArray[float64]
    element_densities: NDArray[float64]

    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict[str, float] = {},
        element_composition: dict[str, float] = {},
        temperature: float = 293.6,
    ):
        super().__init__(name)

        # Temperature
        self.temperature = temperature

        # Check that only one composition is supplied
        if len(nuclide_composition) > 0 and len(element_composition) > 0:
            print_error(
                "Cannot specify both nuclide_composition and element_composition"
            )

        if len(nuclide_composition) == 0 and len(element_composition) == 0:
            print_error(
                "Must specify either nuclide_composition or element_composition"
            )

        # Create lightweight composition objects. Library data and simulation
        # registration are deferred to each object's compilation.
        nearest_temperature = _get_supported_temperature(self.temperature)
        self.nuclide_composition = {
            Nuclide(name, nearest_temperature): density
            for name, density in nuclide_composition.items()
        }
        self.element_composition = {
            Element(name): density for name, density in element_composition.items()
        }

        # Numba representations of the compositions
        self.nuclides = list(self.nuclide_composition)
        self.elements = list(self.element_composition)
        self.nuclide_densities = np.asarray(
            list(self.nuclide_composition.values()), dtype=float
        )
        self.element_densities = np.asarray(
            list(self.element_composition.values()), dtype=float
        )

    def _compile_into_simulation(self, simulation) -> bool:
        """Canonicalize the composition and register it with the simulation."""
        # Skip composition work when this material is already compiled
        if self.compile_ID == simulation.compile_ID:
            return False

        # Resolve composition objects before generic member traversal so only
        # canonical simulation objects are registered.
        nuclide_composition = {}
        element_composition = {}

        # Replace locally created elements with canonical simulation objects
        for element, density in self.element_composition.items():
            element = _get_or_create_element(element.name, simulation, element)
            element_composition[element] = density

        # Replace locally created nuclides with canonical simulation objects
        for nuclide, density in self.nuclide_composition.items():
            nuclide = _get_or_create_nuclide(
                nuclide.name, nuclide.temperature, simulation, nuclide
            )
            nuclide_composition[nuclide] = density

        # Synchronize the packed composition fields with the canonical objects
        self.nuclide_composition = nuclide_composition
        self.element_composition = element_composition
        self.nuclides = list(nuclide_composition)
        self.elements = list(element_composition)
        self.nuclide_densities = np.asarray(
            list(nuclide_composition.values()), dtype=float
        )
        self.element_densities = np.asarray(
            list(element_composition.values()), dtype=float
        )

        # Register the material and compile its canonical composition members
        if not super()._compile_into_simulation(simulation):
            return False

        # Resolve derived properties after nuclide library data has been loaded
        self.fissionable = any(nuclide.fissionable for nuclide in self.nuclides)

        return True

    def __repr__(self):
        text = super().__repr__()

        text += f"  - Temperature: {self.temperature} K\n"
        if len(self.nuclide_composition) > 0:
            text += f"  - Nuclide composition [atoms/barn-cm]\n"
            for nuclide in self.nuclide_composition.keys():
                text += (
                    f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
                )
        if len(self.element_composition) > 0:
            text += f"  - Element composition [atoms/barn-cm]\n"
            for element in self.element_composition.keys():
                text += (
                    f"    - {element.name:<5} | {self.element_composition[element]}\n"
                )
        return text


# Currently supported temperatures
TEMPERATURES = [0.1, 233.15, 273.15, 293.6, 600.0, 900.0, 1200.0, 2500.0]


# ======================================================================================
# Multigroup material
# ======================================================================================


class MaterialMG(MaterialBase):
    """Define a material with multigroup neutron data.

    Parameters
    ----------
    name : str, optional
        User-facing material name.
    capture : ndarray, optional
        Capture macroscopic cross section by incident group.
    scatter : ndarray, optional
        Scattering production matrix in ``[outgoing_group, incident_group]``
        order. Column sums define the scattering cross section.
    fission : ndarray, optional
        Fission macroscopic cross section by incident group.
    nu_s : ndarray, optional
        Mean number of neutrons emitted per scattering event by incident group.
    nu_p : ndarray, optional
        Prompt-fission neutron yield by incident group.
    nu_d : ndarray, optional
        Delayed-fission neutron yield in
        ``[delayed_group, incident_group]`` order.
    chi_p : ndarray, optional
        Prompt-fission spectrum. A one-dimensional spectrum is shared by all
        incident groups; a matrix uses
        ``[outgoing_group, incident_group]`` order.
    chi_d : ndarray, optional
        Delayed-fission spectrum in ``[outgoing_group, delayed_group]`` order.
    speed : ndarray, optional
        Particle speed by energy group.
    decay_rate : ndarray, optional
        Delayed-neutron precursor decay rate by delayed group.

    Notes
    -----
    At least one of ``capture``, ``scatter``, or ``fission`` is required and
    determines the number of energy groups. Cross sections are expected in
    inverse centimetres.

    Examples
    --------
    Define a one-group purely absorbing material:

    >>> import numpy as np
    >>> import mcdc
    >>> absorber = mcdc.MaterialMG(
    ...     name="Absorber",
    ...     capture=np.array([1.0]),
    ... )

    Define a two-group scattering material:

    >>> scatterer = mcdc.MaterialMG(
    ...     name="Scatterer",
    ...     capture=np.array([0.05, 0.10]),
    ...     scatter=np.array([
    ...         [0.70, 0.10],
    ...         [0.20, 0.50],
    ...     ]),
    ...     nu_s=np.array([1.0, 1.0]),
    ... )

    Define a one-group prompt-fission material:

    >>> fuel = mcdc.MaterialMG(
    ...     name="Fuel",
    ...     capture=np.array([0.10]),
    ...     fission=np.array([0.20]),
    ...     nu_p=np.array([2.50]),
    ... )
    """

    # MC/DC framework metadata
    label = "multigroup_material"
    sub_type = MATERIAL_MG

    G: int
    J: int
    mgxs_speed: Annotated[NDArray[float64], ("G",)]
    mgxs_decay_rate: Annotated[NDArray[float64], ("J",)]
    mgxs_capture: Annotated[NDArray[float64], ("G",)]
    mgxs_scatter: Annotated[NDArray[float64], ("G",)]
    mgxs_fission: Annotated[NDArray[float64], ("G",)]
    mgxs_total: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_s: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_p: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_d: Annotated[NDArray[float64], ("G", "J")]
    mgxs_nu_d_total: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_f: Annotated[NDArray[float64], ("G",)]
    mgxs_chi_s: Annotated[NDArray[float64], ("G", "G")]
    mgxs_chi_p: Annotated[NDArray[float64], ("G", "G")]
    mgxs_chi_d: Annotated[NDArray[float64], ("J", "G")]

    def __init__(
        self,
        name: str = "",
        capture: NDArray[float64] | NoneType = None,
        scatter: NDArray[float64] | NoneType = None,
        fission: NDArray[float64] | NoneType = None,
        nu_s: NDArray[float64] | NoneType = None,
        nu_p: NDArray[float64] | NoneType = None,
        nu_d: NDArray[float64] | NoneType = None,
        chi_p: NDArray[float64] | NoneType = None,
        chi_d: NDArray[float64] | NoneType = None,
        speed: NDArray[float64] | NoneType = None,
        decay_rate: NDArray[float64] | NoneType = None,
    ):
        super().__init__(name)

        # Energy group size
        if capture is not None:
            G = len(capture)
        elif scatter is not None:
            G = len(scatter)
        elif fission is not None:
            G = len(fission)
        else:
            G = 0
            print_error("Need to supply capture, scatter, or fission for MaterialMG")
        self.G = G

        # Delayed group size
        J = 0
        if nu_d is not None:
            J = len(nu_d)
        self.J = J

        # Allocate the attributes
        self.mgxs_speed = np.ones(G)
        self.mgxs_decay_rate = np.ones(J) * np.inf
        self.mgxs_capture = np.zeros(G)
        self.mgxs_scatter = np.zeros(G)
        self.mgxs_fission = np.zeros(G)
        self.mgxs_total = np.zeros(G)
        self.mgxs_nu_s = np.ones(G)
        self.mgxs_nu_p = np.zeros(G)
        self.mgxs_nu_d = np.zeros([G, J])
        self.mgxs_nu_d_total = np.zeros([G])
        self.mgxs_nu_f = np.zeros(G)
        self.mgxs_chi_s = np.zeros([G, G])
        self.mgxs_chi_p = np.zeros([G, G])
        self.mgxs_chi_d = np.zeros([J, G])

        # Speed (vector of size G)
        if speed is not None:
            self.mgxs_speed = speed

        # Decay constant (vector of size J)
        if decay_rate is not None:
            self.mgxs_decay_rate = decay_rate

        # Cross-sections (vector of size G)
        if capture is not None:
            self.mgxs_capture = capture
        if scatter is not None:
            self.mgxs_scatter = np.sum(scatter, 0)
        if fission is not None:
            self.mgxs_fission = fission
            self.fissionable = True
        self.mgxs_total = self.mgxs_capture + self.mgxs_scatter + self.mgxs_fission

        # Scattering multiplication (vector of size G)
        if nu_s is not None:
            self.mgxs_nu_s = nu_s

        # Check if nu_p or nu_d is not provided, give fission
        if fission is not None:
            if nu_p is None and nu_d is None:
                print_error("Need to supply nu_p or nu_d for fissionable MaterialMG")

        # Prompt fission production (vector of size G)
        if nu_p is not None:
            self.mgxs_nu_p = nu_p

        # Delayed fission production (matrix of size GxJ)
        if nu_d is not None:
            # Transpose: [dg, gin] -> [gin, dg]
            self.mgxs_nu_d = np.swapaxes(nu_d, 0, 1)[:, :]
        self.mgxs_nu_d_total = np.sum(self.mgxs_nu_d, axis=1)

        # Total fission production (vector of size G)
        self.mgxs_nu_f = np.zeros_like(self.mgxs_nu_p)
        self.mgxs_nu_f += self.mgxs_nu_p
        for j in range(J):
            self.mgxs_nu_f += self.mgxs_nu_d[:, j]

        # Scattering spectrum (matrix of size GxG)
        if scatter is not None:
            # Transpose: [gout, gin] -> [gin, gout]
            self.mgxs_chi_s = np.swapaxes(scatter, 0, 1)[:, :]
            for g in range(G):
                if self.mgxs_scatter[g] > 0.0:
                    self.mgxs_chi_s[g, :] /= self.mgxs_scatter[g]

        # Prompt fission spectrum (matrix of size GxG)
        if nu_p is not None:
            if G == 1:
                self.mgxs_chi_p[:, :] = np.array([[1.0]])
            elif chi_p is None:
                print_error("Need to supply chi_p if nu_p is provided and G > 1")
            else:
                # Convert 1D spectrum to 2D
                if chi_p.ndim == 1:
                    tmp = np.zeros((G, G))
                    for g in range(G):
                        tmp[:, g] = chi_p
                    chi_p = tmp
                # Transpose: [gout, gin] -> [gin, gout]
                self.mgxs_chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
                # Normalize
                for g in range(G):
                    if np.sum(self.mgxs_chi_p[g, :]) > 0.0:
                        self.mgxs_chi_p[g, :] /= np.sum(self.mgxs_chi_p[g, :])

        # Delayed fission spectrum (matrix of size JxG)
        if nu_d is not None:
            if G == 1:
                self.mgxs_chi_d = np.ones([J, G])
            else:
                if chi_d is None:
                    print_error("Need to supply chi_d if nu_d is provided and G > 1")
                else:
                    # Transpose: [gout, dg] -> [dg, gout]
                    self.mgxs_chi_d = np.swapaxes(chi_d, 0, 1)[:, :]
            # Normalize
            for dg in range(J):
                if np.sum(self.mgxs_chi_d[dg, :]) > 0.0:
                    self.mgxs_chi_d[dg, :] /= np.sum(self.mgxs_chi_d[dg, :])

    def __repr__(self):
        text = super().__repr__()

        text += f"  - Multigroup data\n"
        text += f"    - G: {self.G}\n"
        text += f"    - J: {self.J}\n"
        text += f"    - Sigma_c {print_1d_array(self.mgxs_capture)}\n"
        text += f"    - Sigma_s {print_1d_array(self.mgxs_scatter)}\n"
        text += f"    - Sigma_f {print_1d_array(self.mgxs_fission)}\n"
        text += f"    - nu_s {print_1d_array(self.mgxs_nu_s)}\n"
        text += f"    - nu_p {print_1d_array(self.mgxs_nu_p)}\n"
        text += f"    - nu_d {print_1d_array(self.mgxs_nu_d.flatten())}\n"
        text += f"    - chi_s {print_1d_array(self.mgxs_chi_s.flatten())}\n"
        text += f"    - chi_fp {print_1d_array(self.mgxs_chi_p.flatten())}\n"
        text += f"    - chi_fd {print_1d_array(self.mgxs_chi_d.flatten())}\n"
        text += f"    - speed {print_1d_array(self.mgxs_speed)}\n"
        text += f"    - lambda {print_1d_array(self.mgxs_decay_rate)}\n"
        return text


def set_nuclides_from_elements(material, simulation):
    """Expand an elemental composition and register its natural isotopes."""

    material.nuclides = []
    material.nuclide_composition = {}
    nuclide_densities = []

    # Get supported temperature
    nearest_temperature = _get_supported_temperature(material.temperature)

    for element, element_density in material.element_composition.items():
        # To make sure that the abundance is normalized
        norm = 0.0
        for abundance in ISOTOPIC_ABUNDANCE[element.name].values():
            norm += abundance

        # Loop over the nuclide composition
        for nuclide_name, abundance in ISOTOPIC_ABUNDANCE[element.name].items():
            nuclide = _get_or_create_nuclide(
                nuclide_name, nearest_temperature, simulation
            )
            nuclide._compile_into_simulation(simulation)

            # Calculate nuclide density
            nuclide_density = element_density * abundance / norm

            # Register the nuclide composition
            material.nuclides.append(nuclide)
            nuclide_densities.append(nuclide_density)
            material.nuclide_composition[nuclide] = nuclide_density

    material.nuclide_densities = np.array(nuclide_densities)


def set_elements_from_nuclides(material, simulation):
    """Collapse a nuclide composition and register its elements."""

    material.elements = []
    material.element_composition = {}

    # Get the list of the element names
    element_names = []
    for nuclide in material.nuclides:
        element_name = nuclide.name[:2]
        if element_name[1].isdigit():
            element_name = element_name[0]
        if element_name not in element_names:
            element_names.append(element_name)
    element_densities = np.zeros(len(element_names))

    # Iterate over all named elements
    for i, element_name in enumerate(element_names):
        element = _get_or_create_element(element_name, simulation)
        element._compile_into_simulation(simulation)

        material.elements.append(element)

        # Iterate over all nuclides to get the total density
        density = 0.0
        for nuclide, nuclide_density in material.nuclide_composition.items():
            # Skip if non-isotope
            if nuclide.name[: len(element_name)] != element_name:
                continue

            # Accumulate density
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
    """Update a material's fissionable flag from its constituent nuclides."""

    material.fissionable = False
    for nuclide in material.nuclides:
        if nuclide.fissionable:
            material.fissionable = True
            break
