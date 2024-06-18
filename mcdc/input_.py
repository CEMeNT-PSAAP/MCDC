"""
This module contains functions for setting MC/DC input deck.
Docstrings use NumPy formatting.
"""

# Instantiate and get the global variable container
import mcdc.global_ as global_

import h5py, math, mpi4py, os
import numpy as np
import scipy as sp

from mcdc.card import (
    NuclideCard,
    MaterialCard,
    RegionCard,
    SurfaceCard,
    CellCard,
    UniverseCard,
    LatticeCard,
    SourceCard,
)
from mcdc.constant import (
    GYRATION_RADIUS_ALL,
    GYRATION_RADIUS_INFINITE_X,
    GYRATION_RADIUS_INFINITE_Y,
    GYRATION_RADIUS_INFINITE_Z,
    GYRATION_RADIUS_ONLY_X,
    GYRATION_RADIUS_ONLY_Y,
    GYRATION_RADIUS_ONLY_Z,
    PCT_NONE,
    PCT_COMBING,
    PCT_COMBING_WEIGHT,
    INF,
    PI,
    SHIFT,
    REGION_ALL,
)
from mcdc.print_ import print_error
import mcdc.type_ as type_


def nuclide(
    capture=None,
    scatter=None,
    fission=None,
    nu_s=None,
    nu_p=None,
    nu_d=None,
    chi_p=None,
    chi_d=None,
    speed=None,
    decay=None,
    sensitivity=False,
    dsm_Np=1.0,
):
    """
    Create a nuclide

    Parameters
    ----------
    capture : numpy.ndarray (1D), optional
        Capture microscopic cross-section [barn].
    scatter : numpy.ndarray (2D), optional
        Differential scattering microscopic cross-section [gout, gin] [barn].
    fission : numpy.ndarray (1D), optional
        Fission microscopic cross-section [barn].
    nu_s : numpy.ndarray (1D), optional
        Scattering multiplication.
    nu_p : numpy.ndarray (1D), optional
        Prompt fission neutron yield.
    nu_d : numpy.ndarray (2D), optional
        Delayed neutron precursor yield [dg, gin].
    chi_p : numpy.ndarray (2D), optional
        Prompt fission spectrum [gout, gin].
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg].
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s].
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s].
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the nuclide.
    dsm_Np : float
        Average number of derivative particles produced at each
        sensitivity nuclide collision.

    Returns
    -------
    NuclideCard
        The nuclide

    Notes
    -----
    Parameters are set to zeros by default. Energy group size G is determined by the
    size of `capture`, `scatter`, or `fission`. Thus, at least `capture`, `scatter`,
    or `fission` needs to be provided. `nu_p` or `nu_d` is needed if `fission` is
    provided. `chi_p` and `chi_d` are needed if `nu_p` and `nu_d` are provided,
    respectively, and G > 1. Delayed neutron precursor group size J is determined by
    the size of `nu_d`; if `nu_d` is not given, J = 0.

    See also
    --------
    mcdc.material : A material can be defined as a collection of nuclides.
    """
    # Energy group size
    if capture is not None:
        G = len(capture)
    elif scatter is not None:
        G = len(scatter)
    elif fission is not None:
        G = len(fission)
    else:
        print_error("Need to supply capture, scatter, or fission to mcdc.nuclide")

    # Delayed group size
    J = 0
    if nu_d is not None:
        J = len(nu_d)

    # Make nuclide card
    card = NuclideCard(G, J)

    # Set ID
    card.ID = len(global_.input_deck.nuclides)

    # Speed (vector of size G)
    if speed is not None:
        card.speed[:] = speed[:]

    # Decay constant (vector of size J)
    if decay is not None:
        card.decay[:] = decay[:]

    # Cross-sections (vector of size G)
    if capture is not None:
        card.capture[:] = capture[:]
    if scatter is not None:
        card.scatter[:] = np.sum(scatter, 0)[:]
    if fission is not None:
        card.fission[:] = fission[:]
        card.fissionable = True
    card.total[:] = card.capture + card.scatter + card.fission

    # Scattering multiplication (vector of size G)
    if nu_s is not None:
        card.nu_s[:] = nu_s[:]

    # Check if nu_p or nu_d is not provided, give fission
    if fission is not None:
        if nu_p is None and nu_d is None:
            print_error("Need to supply nu_p or nu_d for fissionable mcdc.nuclide")

    # Prompt fission production (vector of size G)
    if nu_p is not None:
        card.nu_p[:] = nu_p[:]

    # Delayed fission production (matrix of size GxJ)
    if nu_d is not None:
        # Transpose: [dg, gin] -> [gin, dg]
        card.nu_d[:, :] = np.swapaxes(nu_d, 0, 1)[:, :]

    # Total fission production (vector of size G)
    card.nu_f += card.nu_p
    for j in range(J):
        card.nu_f += card.nu_d[:, j]

    # Scattering spectrum (matrix of size GxG)
    if scatter is not None:
        # Transpose: [gout, gin] -> [gin, gout]
        card.chi_s[:, :] = np.swapaxes(scatter, 0, 1)[:, :]
        for g in range(G):
            if card.scatter[g] > 0.0:
                card.chi_s[g, :] /= card.scatter[g]

    # Prompt fission spectrum (matrix of size GxG)
    if nu_p is not None:
        if G == 1:
            card.chi_p[:, :] = np.array([[1.0]])
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
            card.chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
            # Normalize
            for g in range(G):
                if np.sum(card.chi_p[g, :]) > 0.0:
                    card.chi_p[g, :] /= np.sum(card.chi_p[g, :])

    # Delayed fission spectrum (matrix of size JxG)
    if nu_d is not None:
        if G == 1:
            card.chi_d[:, :] = np.ones([J, G])
        else:
            if chi_d is None:
                print_error("Need to supply chi_d if nu_d is provided  and G > 1")
            # Transpose: [gout, dg] -> [dg, gout]
            card.chi_d[:, :] = np.swapaxes(chi_d, 0, 1)[:, :]
        # Normalize
        for dg in range(J):
            if np.sum(card.chi_d[dg, :]) > 0.0:
                card.chi_d[dg, :] /= np.sum(card.chi_d[dg, :])

    # Sensitivity setup
    if sensitivity:
        # Set flag
        card.sensitivity = True
        global_.input_deck.technique["sensitivity"] = True
        global_.input_deck.technique["weighted_emission"] = False

        # Set ID
        global_.input_deck.setting["N_sensitivity"] += 1
        card.sensitivity_ID = global_.input_deck.setting["N_sensitivity"]

        # Set dsm_Np
        card.dsm_Np = dsm_Np

    # Add to deck
    global_.input_deck.nuclides.append(card)

    return card


def material(
    nuclides=None,
    capture=None,
    scatter=None,
    fission=None,
    nu_s=None,
    nu_p=None,
    nu_d=None,
    chi_p=None,
    chi_d=None,
    speed=None,
    decay=None,
    sensitivity=False,
    dsm_Np=1.0,
):
    """
    Create a material

    A material is defined either as a collection of nuclides or directly by its
    macroscopic constants.

    Parameters
    ----------
    nuclides : list of tuple of (dictionary, float), optional
        List of pairs of nuclide card and its density [/barn-cm].
    capture : numpy.ndarray (1D), optional
        Capture macroscopic cross-section [/cm].
    scatter : numpy.ndarray (2D), optional
        Differential scattering macroscopic cross-section [gout, gin] [/cm].
    fission : numpy.ndarray (1D), optional
        Fission macroscopic cross-section [/cm].
    nu_s : numpy.ndarray (1D), optional
        Scattering multiplication.
    nu_p : numpy.ndarray (1D), optional
        Prompt fission neutron yield.
    nu_d : numpy.ndarray (2D), optional
        Delayed neutron precursor yield [dg, gin].
    chi_p : numpy.ndarray (2D), optional
        Prompt fission spectrum [gout, gin].
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg].
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s].
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s].
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the material
        (only relevant for single-nuclide material).
    dsm_Np : float
        Average number of derivative particles produced at each
        sensitivity material collision (only relevant for single_nuclide material).

    Returns
    -------
    MaterialCard
        The material

    See also
    --------
    mcdc.nuclide : A material can be defined as a collection of nuclides.
    """
    # If nuclides are not given, and macroscopic constants are given instead,
    # create a nuclide card and set a single-nuclide material
    if nuclides is None:
        card_nuclide = nuclide(
            capture,
            scatter,
            fission,
            nu_s,
            nu_p,
            nu_d,
            chi_p,
            chi_d,
            speed,
            decay,
            sensitivity,
            dsm_Np,
        )
        nuclides = [[card_nuclide, 1.0]]

    # Number of nuclides
    N_nuclide = len(nuclides)

    # Continuous energy mode?
    if isinstance(nuclides[0][0], str):
        global_.input_deck.setting["mode_CE"] = True
        global_.input_deck.setting["mode_MG"] = False

        # Make material card
        card = MaterialCard(N_nuclide)

        # Set ID
        card.ID = len(global_.input_deck.materials)

        # Set the nuclides
        for i in range(N_nuclide):
            nuc_name = nuclides[i][0]
            density = nuclides[i][1]
            if not nuclide_registered(nuc_name):
                nuc_card = NuclideCard(name=nuc_name)

                # Set ID
                nuc_card.ID = len(global_.input_deck.nuclides)

                dir_name = os.getenv("MCDC_XSLIB")
                if dir_name == None:
                    print_error(
                        "Continuous energy data directory not configured \n"+
                        "see https://cement-psaapgithubio.readthedocs.io/en/latest/install.html#configuring-continuous-energy-library \n"
                    )
                with h5py.File(dir_name + "/" + nuc_name + ".h5", "r") as f:
                    if max(f["fission"][:]) > 0.0:
                        nuc_card.fissionable = True

                global_.input_deck.nuclides.append(nuc_card)
            else:
                nuc_card = get_nuclide(nuc_name)
            card.nuclide_IDs[i] = nuc_card.ID
            card.nuclide_densities[i] = density

        # Check if identical material already exists
        identical_material = get_identical_material(card)
        if identical_material is None:
            global_.input_deck.materials.append(card)
            return card
        else:
            return identical_material

    # Nuclide and group sizes
    G = nuclides[0][0].G
    J = nuclides[0][0].J

    # Make material card
    card = MaterialCard(N_nuclide, G, J)

    # Set ID
    card.ID = len(global_.input_deck.materials)

    # Calculate basic XS and determine sensitivity flag
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card.nuclide_IDs[i] = nuc.ID
        card.nuclide_densities[i] = density

        card.capture += nuc.capture * density
        card.scatter += nuc.scatter * density
        card.fission += nuc.fission * density
        card.total += nuc.total * density
        card.sensitivity += nuc.sensitivity * density
    card.sensitivity = bool(card.sensitivity)

    # Calculate effective speed
    # Current approach: weighted by nuclide macroscopic total cross section
    # TODO: other more appropriate way?
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card.speed += nuc.speed * nuc.total * density
    # If vacuum material, just pick the last nuclide
    if max(card.total) == 0.0:
        card.speed[:] = nuc.speed
    else:
        card.speed /= card.total

    # Calculate effective spectra and multiplicities of scattering and prompt fission
    if max(card.scatter) > 0.0:
        nuSigmaS = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaS = np.diag(nuc.scatter) * density
            nu_s = np.diag(nuc.nu_s)
            chi_s = np.transpose(nuc.chi_s)
            nuSigmaS += chi_s.dot(nu_s.dot(SigmaS))
        chi_nu_s = nuSigmaS.dot(np.diag(1.0 / card.scatter))
        card.nu_s = np.sum(chi_nu_s, axis=0)
        card.chi_s = np.transpose(chi_nu_s.dot(np.diag(1.0 / card.nu_s)))
    if max(card.fission) > 0.0:
        nuSigmaF = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaF = np.diag(nuc.fission) * density
            nu_p = np.diag(nuc.nu_p)
            chi_p = np.transpose(nuc.chi_p)
            nuSigmaF += chi_p.dot(nu_p.dot(SigmaF))
        chi_nu_p = nuSigmaF.dot(np.diag(1.0 / card.fission))
        card.nu_p = np.sum(chi_nu_p, axis=0)
        card.chi_p = np.transpose(chi_nu_p.dot(np.diag(1.0 / card.nu_p)))

    # Calculate delayed and total fission multiplicities
    if max(card.fission) > 0.0:
        card.nu_f[:] = card.nu_p[:]
        for j in range(J):
            total = np.zeros(G)
            for i in range(N_nuclide):
                nuc = nuclides[i][0]
                density = nuclides[i][1]
                total += nuc.nu_d[:, j] * nuc.fission * density
            card.nu_d[:, j] = total / card.fission
            card.nu_f += card.nu_d[:, j]

    # Add to deck
    global_.input_deck.materials.append(card)

    return card


def surface(type_, bc="interface", sensitivity=False, dsm_Np=1.0, **kw):
    """
    Create a surface to define the region of a cell.

    Parameters
    ----------
    type\_ : {"plane-x", "plane-y", "plane-z", "plane", "cylinder-x", "cylinder-y",
              "cylinder-z", "sphere", "quadric"}
        Surface type.
    bc : {"interface", "vacuum", "reflective"}
        Surface boundary condition.
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the surface position.
    dsm_Np : int
        Average number of derivative particles produced at each
        sensitivity surface crossing.

    Other Parameters
    ----------------
    x : {float, array_like[float]}
        x-position [cm] for `"plane-x"`.
        If a vector is passed, positions of the surface at the times specified by the
        parameter `"t"`.
    y : {float, array_like[float]}
        y-position [cm] for `"plane-y"`.
        If a vector is passed, positions of the surface at the times specified by the
        parameter `t`.
    z : {float, array_like[float]}
        z-position [cm] for `"plane-z"`.
        If a vector is passed, positions of the surface at the times specified by the
        parameter `t`.
    t : {array_like[float]}
        Time to specify the position of `"plane-x"`, `"plane-y"`, or `"plane-z"`.
    center : array_like[float]
        Center point [cm] for `"cylinder-x"` (y,z), `"cylinder-y"` (x,z),
        `"cylinder-z"` (x,y), or `"sphere"` (x,y,z).
    radius : float
        Radius [cm] for `"cylinder-x"`, `"cylinder-y"`, `"cylinder-z"`, and `"sphere"`.
    A, B, C, D : float
        Coefficients [cm] for `"plane"`.
    A, B, C, D, E, F, G, H, I, J : float
        Coefficients [cm] for `"quadric"`.

    Returns
    -------
    SurfaceCard
        The surface card

    See also
    --------
    mcdc.cell : Create a cell whose region is defined by surfaces.
    """
    # Make surface card
    card = SurfaceCard()

    # Set ID
    card.ID = len(global_.input_deck.surfaces)

    # Check if the selected type is supported
    type_ = check_support(
        "surface type",
        type_,
        [
            "plane-x",
            "plane-y",
            "plane-z",
            "plane",
            "cylinder-x",
            "cylinder-y",
            "cylinder-z",
            "sphere",
            "quadric",
        ],
    )
    card.type = type_

    # Boundary condition
    bc = check_support(
        "surface boundary condition",
        bc,
        [
            "interface",
            "vacuum",
            "reflective",
        ],
    )
    card.boundary_type = bc

    # Sensitivity
    if sensitivity:
        # Set flag
        card.sensitivity = True
        global_.input_deck.technique["sensitivity"] = True
        global_.input_deck.technique["weighted_emission"] = False

        # Set ID
        global_.input_deck.setting["N_sensitivity"] += 1
        card.sensitivity_ID = global_.input_deck.setting["N_sensitivity"]

        # Set dsm_Np
        card.dsm_Np = dsm_Np

    # ==========================================================================
    # Surface attributes
    # ==========================================================================
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
    #   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0

    card.type = type_

    # Set up surface attributes
    if type_ == "plane-x":
        check_requirement("surface plane-x", kw, ["x"])
        card.G = 1.0
        card.linear = True
        if type(kw.get("x")) in [type([]), type(np.array([]))]:
            set_J(kw.get("x"), kw.get("t"), card)
        else:
            card.J[0, 0] = -kw.get("x")
    elif type_ == "plane-y":
        check_requirement("surface plane-y", kw, ["y"])
        card.H = 1.0
        card.linear = True
        if type(kw.get("y")) in [type([]), type(np.array([]))]:
            set_J(kw.get("y"), kw.get("t"), card)
        else:
            card.J[0, 0] = -kw.get("y")
    elif type_ == "plane-z":
        check_requirement("surface plane-z", kw, ["z"])
        card.I = 1.0
        card.linear = True
        if type(kw.get("z")) in [type([]), type(np.array([]))]:
            set_J(kw.get("z"), kw.get("t"), card)
        else:
            card.J[0, 0] = -kw.get("z")
    elif type_ == "plane":
        check_requirement("surface plane", kw, ["A", "B", "C", "D"])
        card.G = kw.get("A")
        card.H = kw.get("B")
        card.I = kw.get("C")
        card.J[0, 0] = kw.get("D")
        card.linear = True
    elif type_ == "cylinder-x":
        check_requirement("surface cylinder-x", kw, ["center", "radius"])
        y, z = kw.get("center")[:]
        r = kw.get("radius")
        card.B = 1.0
        card.C = 1.0
        card.H = -2.0 * y
        card.I = -2.0 * z
        card.J[0, 0] = y**2 + z**2 - r**2
    elif type_ == "cylinder-y":
        check_requirement("surface cylinder-y", kw, ["center", "radius"])
        x, z = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.C = 1.0
        card.G = -2.0 * x
        card.I = -2.0 * z
        card.J[0, 0] = x**2 + z**2 - r**2
    elif type_ == "cylinder-z":
        check_requirement("surface cylinder-z", kw, ["center", "radius"])
        x, y = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.B = 1.0
        card.G = -2.0 * x
        card.H = -2.0 * y
        card.J[0, 0] = x**2 + y**2 - r**2
    elif type_ == "sphere":
        check_requirement("surface sphere", kw, ["center", "radius"])
        x, y, z = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.B = 1.0
        card.C = 1.0
        card.G = -2.0 * x
        card.H = -2.0 * y
        card.I = -2.0 * z
        card.J[0, 0] = x**2 + y**2 + z**2 - r**2
    elif type_ == "quadric":
        check_requirement(
            "surface quadric", kw, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        )
        card.A = kw.get("A")
        card.B = kw.get("B")
        card.C = kw.get("C")
        card.D = kw.get("D")
        card.E = kw.get("E")
        card.F = kw.get("F")
        card.G = kw.get("G")
        card.H = kw.get("H")
        card.I = kw.get("I")
        card.J[0, 0] = kw.get("J")

    # Set normal vector if linear
    if card.linear:
        nx = card.G
        ny = card.H
        nz = card.I
        # Normalize
        norm = (nx**2 + ny**2 + nz**2) ** 0.5
        card.nx = nx / norm
        card.ny = ny / norm
        card.nz = nz / norm

    # Add to deck
    global_.input_deck.surfaces.append(card)

    return card


def set_J(x, t, card):
    # Edit and add the edges
    t[0] = -SHIFT
    t = np.append(t, INF)
    x = np.append(x, x[-1])

    # Reset the constants
    card.J = np.zeros([0, 2])
    card.t = np.array([-SHIFT])

    # Iterate over inputs
    idx = 0
    for i in range(len(t) - 1):
        # Skip if step
        if t[i] == t[i + 1]:
            continue

        # Calculate constants
        J0 = x[i]
        J1 = (x[i + 1] - x[i]) / (t[i + 1] - t[i])

        # Append to surface
        card.J = np.append(card.J, [[J0, J1]], axis=0)
        card.t = np.append(card.t, t[i + 1])

    card.J *= -1
    card.N_slice = len(card.J)


def cell(region=None, fill=None, translation=(0.0, 0.0, 0.0)):
    """
    Create a cell as model building block.

    Parameters
    ----------
    region : RegionCard
        Region that defines the cell geometry.
    fill : MaterialCard or UniverseCard or LatticeCard
        Material/universe/lattice that fills the cell.
    translation : array_like[float], optional
        To translate the origin of the fill (if universe or lattice).

    Returns
    -------
    CellCard
        The cell card.

    See also
    --------
    mcdc.surface : Create a surface to define the region of a cell.
    mcdc.material : Create a material to fill a cell.
    mcdc.universe : Create a universe to fill a cell.
    mcdc.lattice : Create a lattice to fill a cell.
    """

    # Make cell card
    card = CellCard()

    # Set ID
    card.ID = len(global_.input_deck.cells)

    # If region is not assigned, create a region that encompass all
    if region is None:
        region = RegionCard("all")
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)

    # Assign region
    card.region_ID = region.ID

    # Assign fill type and ID
    if fill.tag == "Material":
        card.fill_type = "material"
    elif fill.tag == "Universe":
        card.fill_type = "universe"
    elif fill.tag == "Lattice":
        card.fill_type = "lattice"
    card.fill_ID = fill.ID

    # Translation
    card.translation[:] = translation

    # Get all surface IDs
    surface_IDs = []

    region = global_.input_deck.regions[card.region_ID]
    get_all_surface_IDs(region, surface_IDs)
    surface_IDs = list(set(surface_IDs))

    card.N_surface = len(surface_IDs)
    card.surface_IDs = np.array(surface_IDs)

    # Add to deck
    global_.input_deck.cells.append(card)

    return card


def get_all_surface_IDs(region, surface_IDs):
    if region.type == "halfspace":
        surface = global_.input_deck.surfaces[region.A]
        surface_IDs.append(surface.ID)

    elif region.type in ["intersection", "union"]:
        region1 = global_.input_deck.regions[region.A]
        region2 = global_.input_deck.regions[region.B]

        get_all_surface_IDs(region1, surface_IDs)
        get_all_surface_IDs(region2, surface_IDs)

    elif region.type in ["complement"]:
        region1 = global_.input_deck.regions[region.A]

        get_all_surface_IDs(region1, surface_IDs)


def universe(cells, root=False):
    """
    Define a list of cells as a universe.

    Parameters
    ----------
    cells : list of CellCard
        List of cells that comprise the universe.
    root : bool
        Flag to edit the root universe

    Returns
    -------
    UniverseCard
        The universe card

    See also
    --------
    mcdc.cell : Creates a cell that can be used to define a universe.
    """
    N_cell = len(cells)

    # Edit root universe
    if root:
        # Create and replace placeholder if root is not yet created
        if global_.input_deck.universes[0] == None:
            card = UniverseCard(N_cell)
            card.ID = 0
            global_.input_deck.universes[0] = card
        else:
            card = global_.input_deck.universes[0]

    # Create new universe
    else:
        card = UniverseCard(N_cell)
        card.ID = len(global_.input_deck.universes)

    # Cells
    for i in range(N_cell):
        card.cell_IDs[i] = cells[i].ID

    # Push card
    if not root:
        global_.input_deck.universes.append(card)

    return card


def lattice(x=None, y=None, z=None, universes=None):
    """
    Create a lattice card.

    Parameters
    ----------
    x : array_like[float], optional
        x-coordinates that define the lattice mesh (default None).
    y : array_like[float], optional
        y-coordinates that define the lattice mesh (default None).
    z : array_like[float], optional
        z-coordinates that define the lattice mesh (default None).
    universes : list of (list of dictionary), optional
        List of lists of universe cards that fill the lattice (default None).

    Returns
    -------
    dictionary
        Lattice card.
    """
    # Make lattice card
    card = LatticeCard()
    card.ID = len(global_.input_deck.lattices)

    # Set mesh
    if x is not None:
        card.mesh["x0"] = x[0]
        card.mesh["dx"] = x[1]
        card.mesh["Nx"] = x[2]
    if y is not None:
        card.mesh["y0"] = y[0]
        card.mesh["dy"] = y[1]
        card.mesh["Ny"] = y[2]
    if z is not None:
        card.mesh["z0"] = z[0]
        card.mesh["dz"] = z[1]
        card.mesh["Nz"] = z[2]

    # Set universe IDs
    get_ID = np.vectorize(lambda obj: obj.ID)
    universe_IDs = get_ID(universes)
    ax_expand = []
    if x is None:
        ax_expand.append(2)
    if y is None:
        ax_expand.append(1)
    if z is None:
        ax_expand.append(0)
    for ax in ax_expand:
        universe_IDs = np.expand_dims(universe_IDs, axis=ax)

    # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
    tmp = np.transpose(universe_IDs)
    tmp = np.flip(tmp, axis=1)
    card.universe_IDs = np.flip(tmp, axis=2)

    # Push card
    global_.input_deck.lattices.append(card)
    return card


def source(**kw):
    """
    Create a source card.

    Other Parameters
    ----------------
    point : array_like
        [x, y, z] point position for point source.
    x : array_like
        [x_min and x_max] for uniform source.
    y : array_like
        [y_min and y_max] for uniform source.
    z : array_like
        [z_min and z_max] for uniform source.
    isotropic : bool
        Flag for whether source is isotropic.
    direction : array_like
        [ux, uy, uz] unit vector for parallel beam source.
    white_direction : array_like
        [nx, ny, nz] unit vector of the normal outward direction of the surface
        at which isotropic surface source is emitted. Note that it is similar to the
        mechanics of the typical white boundary condition in reactor physics.
    energy : array_like
        [MG] Probability mass function of the energy group for multigroup source.
        [CE] 2D array of piecewise linear pdf [eV, value].
    time : array_like
        [t_min and t_max] in/at which source is emitted.
    prob : float
        Relative probability (or strength) of the source.

    Returns
    -------
    dictionary
        A source card.
    """
    # Check the supplied keyword arguments
    for key in kw.keys():
        check_support(
            "source parameter",
            key,
            [
                "point",
                "x",
                "y",
                "z",
                "isotropic",
                "direction",
                "white_direction",
                "energy",
                "time",
                "prob",
            ],
            False,
        )

    # Get keyword arguments
    point = kw.get("point")
    x = kw.get("x")
    y = kw.get("y")
    z = kw.get("z")
    isotropic = kw.get("isotropic")
    direction = kw.get("direction")
    white = kw.get("white_direction")
    energy = kw.get("energy")
    time = kw.get("time")
    prob = kw.get("prob")

    # Make source card
    card = SourceCard()

    # Set ID
    card.ID = len(global_.input_deck.sources)

    # Set position
    if point is not None:
        card.x = point[0]
        card.y = point[1]
        card.z = point[2]
    else:
        card.box = True
        if x is not None:
            card.box_x = np.array(x)
        if y is not None:
            card.box_y = np.array(y)
        if z is not None:
            card.box_z = np.array(z)

    # Set direction
    if white is not None:
        card.isotropic = False
        card.white = True
        ux = white[0]
        uy = white[1]
        uz = white[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card.white_x = ux / norm
        card.white_y = uy / norm
        card.white_z = uz / norm
    elif direction is not None:
        card.isotropic = False
        ux = direction[0]
        uy = direction[1]
        uz = direction[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card.ux = ux / norm
        card.uy = uy / norm
        card.uz = uz / norm

    # Set energy
    if energy is not None:
        if global_.input_deck.setting["mode_MG"]:
            group = np.array(energy)
            # Normalize
            card.group = group / np.sum(group)
        if global_.input_deck.setting["mode_CE"]:
            energy = np.array(energy)
            # Resize
            card.energy = np.zeros(energy.shape)
            # Set energy
            card.energy[0, :] = energy[0, :]
            # Normalize pdf
            card.energy[1, :] = energy[1, :] / np.trapz(energy[1, :], x=energy[0, :])
            # Make cdf
            card.energy[1, :] = sp.integrate.cumulative_trapezoid(
                card.energy[1], x=card.energy[0], initial=0.0
            )
    else:
        # Default for MG
        if global_.input_deck.setting["mode_MG"]:
            G = global_.input_deck.materials[0].G
            group = np.ones(G)
            card.group = group / np.sum(group)
        # Default for CE
        if global_.input_deck.setting["mode_CE"]:
            # Normalize pdf
            card.energy[1, :] = card.energy[1, :] / np.trapz(card.energy[1, :], x=card.energy[0, :])
            # Make cdf
            card.energy[1, :] = sp.integrate.cumulative_trapezoid(
                card.energy[1], x=card.energy[0], initial=0.0
            )

    # Set time
    if time is not None:
        card.time = np.array(time)

    # Set probability
    if prob is not None:
        card.prob = prob

    # Push card
    global_.input_deck.sources.append(card)

    return card


# ==============================================================================
# Setting
# ==============================================================================


def setting(**kw):
    """
    Create a setting card.

    Other Parameters
    ----------------
    N_particle : int
        Number of MC particle histories to run (for k-eigen and iQMC its /iteration).
    N_batch : int
        Number of batches to run.
    rng_seed : int
        Random number seed.
    time_boundary : float
        The time edge of the problem, after which all particles will be killed.
    progress_bar : bool
        Whether to display the progress bar (default True; disable when running MC/DC in a loop).
    caching : bool
        Whether to store or delete compiled Numba kernels (default True will store; False will delete existing __pycache__ folder).
        see :ref:`Caching`.
    output_name : str
        Name of the output file MC/DC should save data in (default "output.h5").
    save_input_deck : bool
        Whether to save the input deck information to the output file (default False).
    particle_tracker : bool
        Whether to track paths of all individual particles histories, memory issues abound (default False).
    k_eff : str
        Whether to run a k-eigenvalue problem.
    source_file : str
        Source file path and name.
    IC_file : str
        Path to a file containing a description of an initial condition.
    active_bank_buff : int
        Size of the activate particle bank buffer, for MPI runs.
    census_bank_buff : int
        Sets size of the census buffer particle bank.

    Returns
    -------
    dictionary
        A setting card.
    """

    # Check the supplied keyword arguments
    for key in kw.keys():
        check_support(
            "setting parameter",
            key,
            [
                "N_particle",
                "N_batch",
                "rng_seed",
                "time_boundary",
                "progress_bar",
                "output_name",
                "save_input_deck",
                "particle_tracker",
                "k_eff",
                "source_file",
                "IC_file",
                "active_bank_buff",
                "census_bank_buff",
                "caching",
            ],
            False,
        )

    # Get keyword arguments
    N_particle = kw.get("N_particle")
    N_batch = kw.get("N_batch")
    rng_seed = kw.get("rng_seed")
    time_boundary = kw.get("time_boundary")
    progress_bar = kw.get("progress_bar")
    output = kw.get("output_name")
    save_input_deck = kw.get("save_input_deck")
    particle_tracker = kw.get("particle_tracker")
    k_eff = kw.get("k_eff")
    source_file = kw.get("source_file")
    IC_file = kw.get("IC_file")
    bank_active_buff = kw.get("active_bank_buff")
    bank_census_buff = kw.get("census_bank_buff")
    caching = kw.get("caching")

    # Check if setting card has been initialized
    card = global_.input_deck.setting

    # Number of particles
    if N_particle is not None:
        card["N_particle"] = int(N_particle)

    # Number of batches
    if N_batch is not None:
        card["N_batch"] = int(N_batch)

    # Time boundary
    if time_boundary is not None:
        card["time_boundary"] = time_boundary

    # RNG seed and stride
    if rng_seed is not None:
        card["rng_seed"] = rng_seed

    # Output .h5 file name
    if output is not None:
        card["output_name"] = output

    # Progress bar
    if progress_bar is not None:
        card["progress_bar"] = progress_bar

    # k effective
    if k_eff is not None:
        card["k_init"] = k_eff

    # Maximum active bank size
    if bank_active_buff is not None:
        card["bank_active_buff"] = int(bank_active_buff)

    # Census bank size multiplier
    if bank_census_buff is not None:
        card["bank_census_buff"] = int(bank_census_buff)

    # caching is normally enabled
    if caching is not None:
        card["caching"] = caching

    # Particle tracker
    if particle_tracker is not None:
        card["track_particle"] = particle_tracker
        if particle_tracker and mpi4py.MPI.COMM_WORLD.Get_size() > 1:
            print_error("Particle tracker currently only runs on a single MPI rank")

    # Save input deck?
    if save_input_deck is not None:
        card["save_input_deck"] = save_input_deck

    # Source file
    if source_file is not None:
        card["source_file"] = True
        card["source_file_name"] = source_file

        # Set number of particles
        card_setting = global_.input_deck.setting
        with h5py.File(source_file, "r") as f:
            card_setting["N_particle"] = f["particles_size"][()]

    # IC file
    if IC_file is not None:
        card["IC_file"] = True
        card["IC_file_name"] = IC_file

        # Set number of particles
        card_setting = global_.input_deck.setting
        with h5py.File(IC_file, "r") as f:
            card_setting["N_particle"] = f["IC/neutrons_size"][()]
            card_setting["N_precursor"] = f["IC/precursors_size"][()]

    # TODO: Allow both source and IC files
    if IC_file and source_file:
        print_error("Using both source and IC files is not supported yet.")


def eigenmode(
    N_inactive=0, N_active=0, k_init=1.0, gyration_radius=None, save_particle=False
):
    """
    Create an eigenmode card.

    Parameters
    ----------
    N_inactive : int
        Number of cycles not included when averaging the k-eigenvalue (default 0).
    N_active : int
        Number of cycles to include for statistics of the k-eigenvalue (default 0).
    k_init : float
        Initial k value to iterate on (default 1.0).
    gyration_radius : float, optional
        Specify a gyration radius (default None).
    save_particle : bool
        Whether particle track outputs in a tally mesh (default False).

    Returns
    -------
    dictionary
        A eigenmode card.
    """

    # Update setting card
    card = global_.input_deck.setting
    card["N_inactive"] = N_inactive
    card["N_active"] = N_active
    card["N_cycle"] = N_inactive + N_active
    card["mode_eigenvalue"] = True
    card["k_init"] = k_init
    card["save_particle"] = save_particle

    # Gyration radius setup
    if gyration_radius is not None:
        card["gyration_radius"] = True
        if gyration_radius == "all":
            card["gyration_radius_type"] = GYRATION_RADIUS_ALL
        elif gyration_radius == "infinite-x":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_X
        elif gyration_radius == "infinite-y":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_Y
        elif gyration_radius == "infinite-z":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_Z
        elif gyration_radius == "only-x":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_X
        elif gyration_radius == "only-y":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_Y
        elif gyration_radius == "only-z":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_Z
        else:
            print_error("Unknown gyration radius type")


# ==============================================================================
# Technique
# ==============================================================================


def implicit_capture():
    """
    Activate implicit capture (implies no weighted emission).
    """
    card = global_.input_deck.technique
    card["implicit_capture"] = True
    card["weighted_emission"] = False


def weighted_emission(flag):
    """
    Activate weighted emission variance reduction technique.

    Parameters
    ----------
    flag : bool
        True to activate weighted emission.
    """

    card = global_.input_deck.technique
    card["weighted_emission"] = flag


def population_control(pct="combing"):
    """
    Set population control techniques.

    Parameters
    ----------
    pct : str, optional
        Population control method (default "combing").
    """

    card = global_.input_deck.technique
    card["population_control"] = True
    card["weighted_emission"] = False
    if pct == "combing":
        card["pct"] = PCT_COMBING
    elif pct == "combing-weight":
        card["pct"] = PCT_COMBING_WEIGHT
    else:
        print_error("Unknown PCT type " + pct)


def branchless_collision():
    """
    Activate branchless collision variance reduction technique (implies no weighted emission).
    """
    card = global_.input_deck.technique
    card["branchless_collision"] = True
    card["weighted_emission"] = False


def time_census(t):
    """
    Set time-census boundaries.

    Parameters
    ----------
    t : array_like[float]
        The time-census boundaries.

    Returns
    -------
        None (in-place card alterations).
    """

    # Remove census beyond the final tally time grid point
    while True:
        if t[-1] >= global_.input_deck.tally["mesh"]["t"][-1]:
            t = t[:-1]
        else:
            break

    # Add the default, final census-at-infinity
    t = np.append(t, INF)

    # Set the time census parameters
    card = global_.input_deck.setting
    card["census_time"] = t
    card["N_census"] = len(t)


def weight_window(x=None, y=None, z=None, t=None, window=None, width=None):
    """
    Activate weight window variance reduction technique.

    Parameters
    ----------
    x : array_like[float], optional
        Location of the weight window in x (default None).
    y : array_like[float], optional
        Location of the weight window in y (default None).
    z : array_like[float], optional
        Location of the weight window in z (default None).
    t : array_like[float], optional
        Location of the weight window in t (default None).
    window : array_like[float], optional
        Bound of the statistic weight of the window (default None).
    width : array_like[float], optional
        Statistical width the window will apply (default None).

    Returns
    -------
        A weight window card.

    """
    card = global_.input_deck.technique
    card["weight_window"] = True

    # Set width
    if width is not None:
        card["ww_width"] = width

    # Set mesh
    if x is not None:
        card["ww_mesh"]["x"] = x
    if y is not None:
        card["ww_mesh"]["y"] = y
    if z is not None:
        card["ww_mesh"]["z"] = z
    if t is not None:
        card["ww_mesh"]["t"] = t

    # Set window
    ax_expand = []
    if t is None:
        ax_expand.append(0)
    if x is None:
        ax_expand.append(1)
    if y is None:
        ax_expand.append(2)
    if z is None:
        ax_expand.append(3)
    window /= np.max(window)
    for ax in ax_expand:
        window = np.expand_dims(window, axis=ax)
    card["ww"] = window

    return card


def domain_decomposition(
    x=None,
    y=None,
    z=None,
    exchange_rate=100000,
    work_ratio=None,
    repro=True,
):
    """
    Activate domain decomposition.

    Parameters
    ----------
    x : array_like[float], optional
        Location of subdomain boundaries in x (default None).
    y : array_like[float], optional
        Location of subdomain boundaries in y (default None).
    z : array_like[float], optional
        Location of subdomain boundaries in z (default None).
    exchange_rate : float, optional
        Number of particles to acumulate in the domain banks before sending.
    work_ratio : array_like[integer], optional
        Number of processors in each domain

    Returns
    -------
        A domain decomposition card.

    """
    card = global_.input_deck.technique
    card["domain_decomposition"] = True
    card["dd_exchange_rate"] = int(exchange_rate)
    card["dd_repro"] = repro
    dom_num = 1
    # Set mesh
    if x is not None:
        card["dd_mesh"]["x"] = x
        dom_num *= len(x)
    if y is not None:
        card["dd_mesh"]["y"] = y
        dom_num *= len(y)
    if z is not None:
        card["dd_mesh"]["z"] = z
        dom_num += len(z)
    # Set work ratio
    if work_ratio is None:
        card["dd_work_ratio"] = None
    elif work_ratio is not None:
        card["dd_work_ratio"] = work_ratio
    card["dd_idx"] = 0
    card["dd_xp_neigh"] = []
    card["dd_xn_neigh"] = []
    card["dd_yp_neigh"] = []
    card["dd_yn_neigh"] = []
    card["dd_zp_neigh"] = []
    card["dd_zn_neigh"] = []
    return card


def iQMC(
    g=None,
    t=None,
    x=None,
    y=None,
    z=None,
    phi0=None,
    source0=None,
    source_x0=None,
    source_y0=None,
    source_z0=None,
    krylov_restart=None,
    fixed_source=None,
    scramble=False,
    maxitt=25,
    tol=1e-6,
    N_dim=6,
    seed=12345,
    preconditioner_sweeps=5,
    generator="halton",
    fixed_source_solver="source_iteration",
    eigenmode_solver="power_iteration",
    score=[],
):
    """
    Set iQMC settings.

    Parameters
    ----------
    g : array_like[float], optional
        Energy values that define energy mesh (default None).
    t : array_like[float], optional
        Time values that define time mesh (default None).
    x : array_like[float], optional
        x-coordinates that define spacial mesh (default None).
    y : array_like[float], optional
        y-coordinates that define spacial mesh (default None).
    z : array_like[float], optional
        z-coordinates that define spacial mesh (default None).
    phi0 : array_like[float], optional
        Initial scalar flux (default None).
    source0 : array_like[float], optional
        Initial particle source (default None).
    source_x0 : array_like[float], optional
        Initial source for tilt-x (default None).
    source_y0 : array_like[float], optional
        Initial source for tilt-y (default None).
    source_z0 : array_like[float], optional
        Initial source for tilt-z (default None).
    krylov_restart : int, optional
        Max number of iterations for Krylov iteration (default same as maxitt).
    fixed_source : array_like[float], optional
        Fixed source (default same as phi0).
    scramble : bool, optional
        Whether to scramble (default False, implies over-easy).
    maxitt : int, optional
        Maximum number of iterations allowed before termination (default 25).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    N_dim : int, optional
        Problem dimensionality (default 6).
    seed : int, optional
        Random number seed (default 12345).
    preconditioner_sweeps : int, optional
        Number of preconditioner sweeps (default 5).
    generator : str, optional
        Low-discrepancy sequence generator (default "halton").
    fixed_source_solver : str, optional
        Deterministic solver for fixed-source problem (default "source_iteration").
    eigenmode_solver : str, optional
        Solver for k-eigenvalue problem (default "power_iteration").
    score : list of str
        List of tally types (default empty list).

    Returns
    -------
        None (in-place card alterations).
    """

    card = global_.input_deck.technique
    card["iQMC"] = True
    card["iqmc"]["tol"] = tol
    card["iqmc"]["maxitt"] = maxitt
    card["iqmc"]["generator"] = generator
    card["iqmc"]["N_dim"] = N_dim
    card["iqmc"]["scramble"] = scramble
    card["iqmc"]["seed"] = seed

    # Set mesh
    if g is not None:
        card["iqmc"]["mesh"]["g"] = g
    if t is not None:
        card["iqmc"]["mesh"]["t"] = t
    if x is not None:
        card["iqmc"]["mesh"]["x"] = x
    if y is not None:
        card["iqmc"]["mesh"]["y"] = y
    if z is not None:
        card["iqmc"]["mesh"]["z"] = z

    ax_expand = []
    if g is None:
        ax_expand.append(0)
    if t is None:
        ax_expand.append(1)
    if x is None:
        ax_expand.append(2)
    if y is None:
        ax_expand.append(3)
    if z is None:
        ax_expand.append(4)
    for ax in ax_expand:
        phi0 = np.expand_dims(phi0, axis=ax)
        if fixed_source is not None:
            fixed_source = np.expand_dims(fixed_source, axis=ax)
        else:
            fixed_source = np.zeros_like(phi0)

    if krylov_restart is None:
        krylov_restart = maxitt

    if source0 is None:
        source0 = np.zeros_like(phi0)

    score_list = card["iqmc"]["score_list"]
    for name in score:
        score_list[name] = True

    if score_list["tilt-x"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_x0 is None:
            source_x0 = np.zeros_like(phi0)

    if score_list["tilt-y"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_y0 is None:
            source_y0 = np.zeros_like(phi0)

    if score_list["tilt-z"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_z0 is None:
            source_z0 = np.zeros_like(phi0)

    card["iqmc"]["score"]["flux"] = phi0
    card["iqmc"]["score"]["tilt-x"] = source_x0
    card["iqmc"]["score"]["tilt-y"] = source_y0
    card["iqmc"]["score"]["tilt-z"] = source_z0
    card["iqmc"]["source"] = source0
    card["iqmc"]["fixed_source"] = fixed_source
    card["iqmc"]["fixed_source_solver"] = fixed_source_solver
    card["iqmc"]["eigenmode_solver"] = eigenmode_solver
    card["iqmc"]["preconditioner_sweeps"] = preconditioner_sweeps
    card["iqmc"]["krylov_restart"] = krylov_restart


def weight_roulette(w_threshold=0.2, w_survive=1.0):
    """
    Activate weight roulette technique.

    If neutron weight is below `w_threshold`, then enter weight roulette
    technique with survival weight `w_survive`.

    Parameters
    ----------
    w_threshold : float
        Weight_roulette() is called on a particle if P['w'] <= wr_threshold.
    w_survive : float
        Weight of surviving particle.

    Returns
    -------
        None (in-place card alterations).
    """
    card = global_.input_deck.technique
    card["weight_roulette"] = True
    card["wr_threshold"] = w_threshold
    card["wr_survive"] = w_survive


def IC_generator(
    N_neutron=0,
    N_precursor=0,
    cycle_stretch=1.0,
    neutron_density=None,
    max_neutron_density=None,
    precursor_density=None,
    max_precursor_density=None,
):
    """
    Activate initial condition generator.

    The initial condition generator samples initial neutrons and precursors
    during an eigenvalue simulation.

    Parameters
    ----------
    N_neutron : int
        Neutron target size.
    N_precursor : int
        Delayed neutron precursor target size.
    cycle_stretch : float
        Factor to stretch number of cycles. Higher cycle stretch reduces inter-cycle
        correlation.
    neutron_density, max_neutron_density : float
        Total and maximum neutron density, required if `N_neutron` > 0.
    precursor_density, max_precursor_density : float
        Total and maximum precursor density, required if `N_precursor` > 0.

    Returns
    -------
        None (in-place card alterations).
    """

    # Turn on eigenmode and population control
    eigenmode()
    population_control()

    # Set parameters
    card = global_.input_deck.technique
    card["IC_generator"] = True
    card["IC_N_neutron"] = N_neutron
    card["IC_N_precursor"] = N_precursor

    # Setting parameters
    card_setting = global_.input_deck.setting
    N_particle = card_setting["N_particle"]

    # Check optional parameters
    if N_neutron > 0.0:
        if neutron_density is None or max_neutron_density is None:
            print_error("IC generator requires neutron_density and max_neutron_density")
        card["IC_neutron_density"] = N_particle * neutron_density
        card["IC_neutron_density_max"] = max_neutron_density
    if N_precursor > 0.0:
        if precursor_density is None:
            print_error(
                "IC generator requires precursor_density and max_precursor_density"
            )
        card["IC_precursor_density"] = N_particle * precursor_density
        card["IC_precursor_density_max"] = max_precursor_density

    # Set number of active cycles
    n = card["IC_neutron_density"]
    n_max = card["IC_neutron_density_max"]
    C = card["IC_precursor_density"]
    C_max = card["IC_precursor_density_max"]
    N_cycle1 = 0.0
    N_cycle2 = 0.0
    if N_neutron > 0:
        N_cycle1 = math.ceil(cycle_stretch * math.ceil(n_max / n * N_neutron))
    if N_precursor > 0:
        N_cycle2 = math.ceil(cycle_stretch * math.ceil(C_max / C * N_precursor))
    N_cycle = max(N_cycle1, N_cycle2)
    card_setting["N_cycle"] = N_cycle
    card_setting["N_active"] = N_cycle


def dsm(order=1):
    """
    Direct sensitivity method

    Parameters
    ----------
    order : int, optional
        order of the sensitivity to probe, by default 1
    """

    card = global_.input_deck.technique
    if order > 2:
        print_error("DSM currently only supports up to second-order sensitivities")
    card["dsm_order"] = order


def uq(**kw):
    """
    Activate uncertainty quantification.

    Other Parameters
    ----------------
    material : dictionary, optional
        Material card of material with uncertain parameters.
    nuclide : dictionary, optional
        Nuclear card of nuclide with uncertain parameters.
    distribution : {"uniform"}
        Probability distribution of uncertain parameters.

    Returns
    -------
        None (in-place card alterations).
    """

    def append_card(delta_card, global_tag):
        delta_card.distribution = dist
        delta_card.flags = []
        for key in kw.keys():
            check_support(parameter.tag + " parameter", key, parameter_list, False)
            delta_card.flags.append(key)
            setattr(delta_card, key, kw[key])
        global_.input_deck.uq_deltas[global_tag].append(delta_card)

    global_.input_deck.technique["uq"] = True
    # Make sure N_batch > 1
    if global_.input_deck.setting["N_batch"] <= 1:
        print_error(
            "Must set N_batch>1 with global_.setting() prior to global_.uq() call."
        )

    # Check uq parameter
    parameter_ = check_support(
        "uq parameter",
        list(kw)[0],
        ["nuclide", "material", "surface", "source"],
        False,
    )
    parameter = kw[parameter_]
    del kw[parameter_]
    parameter.uq = True

    # Confirm supplied distribution
    check_requirement("uq", kw, ["distribution"])
    dist = check_support("distribution", kw["distribution"], ["uniform"], False)
    del kw["distribution"]

    # Only remaining keywords should be the parameter delta(s)

    if parameter.tag == "Material":
        parameter_list = [
            "capture",
            "scatter",
            "fission",
            "nu_s",
            "nu_p",
            "nu_d",
            "chi_p",
            "chi_d",
            "speed",
            "decay",
        ]
        global_tag = "materials"
        if parameter.N_nuclide == 1:
            nuc_card = NuclideCard(parameter.G, parameter.J)
            nuc_card.ID = parameter.nuclide_IDs[0]
            append_card(nuc_card, "nuclides")
        delta_card = MaterialCard(parameter.N_nuclide, parameter.G, parameter.J)
        for name in ["ID", "nuclide_IDs", "nuclide_densities"]:
            setattr(delta_card, name, getattr(parameter, name))
    elif parameter.tag == "Nuclide":
        parameter_list = [
            "capture",
            "scatter",
            "fission",
            "nu_s",
            "nu_p",
            "nu_d",
            "chi_p",
            "chi_d",
            "speed",
            "decay",
        ]
        global_tag = "nuclides"
        delta_card = make_card_nuclide(parameter.G, parameter.J)
        delta_card["ID"] = parameter.ID
    append_card(delta_card, global_tag)
    # elif parameter['tag'] is 'Surface':
    # elif parameter['tag'] is 'Source':


# ==============================================================================
# Util
# ==============================================================================


def nuclide_registered(name):
    for card in global_.input_deck.nuclides:
        if name == card.name:
            return True
    return False


def get_nuclide(name):
    for card in global_.input_deck.nuclides:
        if name == card.name:
            return card


def get_identical_material(card):
    nuclide_IDs = card.nuclide_IDs
    nuclide_densities = card.nuclide_densities
    for material in global_.input_deck.materials:
        if len(nuclide_IDs) == len(material.nuclide_IDs):
            if (nuclide_IDs == material.nuclide_IDs).all() and (nuclide_densities == material.nuclide_densities).all():
                return material
    return None


def print_card(card):
    if isinstance(card, SurfaceHandle):
        card = card.card
    for key in card:
        if key == "tag":
            print(card[key] + " card")
        else:
            print("  " + key + " : " + str(card[key]))


def check_support(label, value, supported, replace=True):
    if replace:
        value = value.replace("_", "-").replace(" ", "-").lower()
    supported_str = "{"
    for str_ in supported:
        supported_str += str_ + ", "
    supported_str = supported_str[:-2] + "}"
    if value not in supported:
        print_error("Unsupported " + label + ": " + value + "\n" + supported_str)
    return value


def check_requirement(label, kw, required):
    missing = "{"
    error = False
    for req in required:
        if req not in kw.keys():
            error = True
            missing += req + ", "
    missing = missing[:-2] + "}"
    if error:
        print_error("Parameters " + missing + " are required for" + label)


# ==============================================================================
# Reset
# ==============================================================================


def reset():
    global_.input_deck.reset()
