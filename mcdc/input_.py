"""
This module contains functions for setting MC/DC input deck.
The input deck class is defined in `card.py` and instantiated in `global_.py`.
"""

import h5py, math, mpi4py, os
import numpy as np
import scipy as sp

import mcdc.type_ as type_

from mcdc.card import (
    SurfaceHandle,
    make_card_nuclide,
    make_card_material,
    make_card_surface,
    make_card_cell,
    make_card_universe,
    make_card_lattice,
    make_card_source,
    make_card_uq,
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
)
from mcdc.print_ import print_error

# Get and rename mcdc global variables
import mcdc.global_ as mcdc


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
    Create a nuclide card.

    Parameters
    ----------
    capture : numpy.ndarray (1D), optional
        Capture microscopic cross-section [barn]
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
        Prompt fission spectrum [gout, gin]
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg]
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s]
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s]
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the nuclide
    dsm_Np : float
        Average number of derivative particles produced at each
        sensitivity nuclide collision

    Returns
    -------
    dictionary
        A nuclide card

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
    card = make_card_nuclide(G, J)
    card["ID"] = len(mcdc.input_deck.nuclides)

    # Speed (vector of size G)
    if speed is not None:
        card["speed"][:] = speed[:]

    # Decay constant (vector of size J)
    if decay is not None:
        card["decay"][:] = decay[:]

    # Cross-sections (vector of size G)
    if capture is not None:
        card["capture"][:] = capture[:]
    if scatter is not None:
        card["scatter"][:] = np.sum(scatter, 0)[:]
    if fission is not None:
        card["fission"][:] = fission[:]
        card["fissionable"] = True
    card["total"][:] = card["capture"] + card["scatter"] + card["fission"]

    # Scattering multiplication (vector of size G)
    if nu_s is not None:
        card["nu_s"][:] = nu_s[:]

    # Check if nu_p or nu_d is not provided, give fission
    if fission is not None:
        if nu_p is None and nu_d is None:
            print_error("Need to supply nu_p or nu_d for fissionable mcdc.nuclide")

    # Prompt fission production (vector of size G)
    if nu_p is not None:
        card["nu_p"][:] = nu_p[:]

    # Delayed fission production (matrix of size GxJ)
    if nu_d is not None:
        # Transpose: [dg, gin] -> [gin, dg]
        card["nu_d"][:, :] = np.swapaxes(nu_d, 0, 1)[:, :]

    # Total fission production (vector of size G)
    card["nu_f"] += card["nu_p"]
    for j in range(J):
        card["nu_f"] += card["nu_d"][:, j]

    # Scattering spectrum (matrix of size GxG)
    if scatter is not None:
        # Transpose: [gout, gin] -> [gin, gout]
        card["chi_s"][:, :] = np.swapaxes(scatter, 0, 1)[:, :]
        for g in range(G):
            if card["scatter"][g] > 0.0:
                card["chi_s"][g, :] /= card["scatter"][g]

    # Prompt fission spectrum (matrix of size GxG)
    if nu_p is not None:
        if G == 1:
            card["chi_p"][:, :] = np.array([[1.0]])
        elif chi_p is None:
            print_error("Need to supply chi_p if nu_p is provided and G > 1")
        else:
            # Transpose: [gout, gin] -> [gin, gout]
            card["chi_p"][:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
            # Normalize
            for g in range(G):
                if np.sum(card["chi_p"][g, :]) > 0.0:
                    card["chi_p"][g, :] /= np.sum(card["chi_p"][g, :])

    # Delayed fission spectrum (matrix of size JxG)
    if nu_d is not None:
        if G == 1:
            card["chi_d"][:, :] = np.ones([J, G])
        else:
            if chi_d is None:
                print_error("Need to supply chi_d if nu_d is provided  and G > 1")
            # Transpose: [gout, dg] -> [dg, gout]
            card["chi_d"][:, :] = np.swapaxes(chi_d, 0, 1)[:, :]
        # Normalize
        for dg in range(J):
            if np.sum(card["chi_d"][dg, :]) > 0.0:
                card["chi_d"][dg, :] /= np.sum(card["chi_d"][dg, :])

    # Sensitivity setup
    if sensitivity:
        # Set flag
        card["sensitivity"] = True
        mcdc.input_deck.technique["sensitivity"] = True
        mcdc.input_deck.technique["weighted_emission"] = False

        # Set ID
        mcdc.input_deck.setting["N_sensitivity"] += 1
        card["sensitivity_ID"] = mcdc.input_deck.setting["N_sensitivity"]

        # Set dsm_Np
        card["dsm_Np"] = dsm_Np

    # Push card
    mcdc.input_deck.nuclides.append(card)
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
    name="P",
    sensitivity=False,
    dsm_Np=1.0,
):
    """
    Create a material card.

    The material card is defined either as a collection of nuclides or directly by its
    macroscopic constants.

    Parameters
    ----------
    nuclides : list of tuple of (dictionary, float), optional
        List of pairs of nuclide card and its density [/barn-cm]
    capture : numpy.ndarray (1D), optional
        Capture macroscopic cross-section [/cm]
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
        Prompt fission spectrum [gout, gin]
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg]
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s]
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s]
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the material
        (only relevant for single-nuclide material)
    dsm_Np : float
        Average number of derivative particles produced at each
        sensitivity material collision (only relevant for single_nuclide material)

    Returns
    -------
    dictionary
        A material card

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
        mcdc.input_deck.setting["mode_CE"] = True
        mcdc.input_deck.setting["mode_MG"] = False

        # Make material card
        card = make_card_material(N_nuclide)
        card["ID"] = len(mcdc.input_deck.materials)

        # Set the nuclides
        for i in range(N_nuclide):
            nuc_name = nuclides[i][0]
            density = nuclides[i][1]
            if not nuclide_registered(nuc_name):
                nuc_ID = len(mcdc.input_deck.nuclides)
                nuc_card = make_card_nuclide()
                nuc_card["name"] = nuc_name
                nuc_card["ID"] = nuc_ID

                dir_name = os.getenv("MCDC_XSLIB")
                with h5py.File(dir_name + "/" + nuc_name + ".h5", "r") as f:
                    if max(f["fission"][:]) > 0.0:
                        nuc_card["fissionable"] = True

                mcdc.input_deck.nuclides.append(nuc_card)
            else:
                nuc_card = get_nuclide(nuc_name)
            card["nuclide_IDs"][i] = nuc_card["ID"]
            card["nuclide_densities"][i] = density

        # Add to deck
        mcdc.input_deck.materials.append(card)
        return card

    # Nuclide and group sizes
    G = nuclides[0][0]["G"]
    J = nuclides[0][0]["J"]

    # Make material card
    card = make_card_material(N_nuclide, G, J)
    card["ID"] = len(mcdc.input_deck.materials)

    if name is not None:
        card["name"] = name
    else:
        card["name"] = card["ID"]

    # Calculate basic XS and determine sensitivity flag
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card["nuclide_IDs"][i] = nuc["ID"]
        card["nuclide_densities"][i] = density
        for tag in ["capture", "scatter", "fission", "total", "sensitivity"]:
            card[tag] += nuc[tag] * density
    card["sensitivity"] = bool(card["sensitivity"])

    # Calculate effective speed
    # Current approach: weighted by nuclide macroscopic total cross section
    # TODO: other more appropriate way?
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card["speed"] += nuc["speed"] * nuc["total"] * density
    # Check if vacuum material
    if max(card["total"]) == 0.0:
        card["speed"][:] = nuc["speed"][:]
    else:
        card["speed"] /= card["total"]

    # Calculate effective spectra and multiplicities of scattering and prompt fission
    if max(card["scatter"]) > 0.0:
        nuSigmaS = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaS = np.diag(nuc["scatter"]) * density
            nu_s = np.diag(nuc["nu_s"])
            chi_s = np.transpose(nuc["chi_s"])
            nuSigmaS += chi_s.dot(nu_s.dot(SigmaS))
        chi_nu_s = nuSigmaS.dot(np.diag(1.0 / card["scatter"]))
        card["nu_s"] = np.sum(chi_nu_s, axis=0)
        card["chi_s"] = np.transpose(chi_nu_s.dot(np.diag(1.0 / card["nu_s"])))
    if max(card["fission"]) > 0.0:
        nuSigmaF = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaF = np.diag(nuc["fission"]) * density
            nu_p = np.diag(nuc["nu_p"])
            chi_p = np.transpose(nuc["chi_p"])
            nuSigmaF += chi_p.dot(nu_p.dot(SigmaF))
        chi_nu_p = nuSigmaF.dot(np.diag(1.0 / card["fission"]))
        card["nu_p"] = np.sum(chi_nu_p, axis=0)
        card["chi_p"] = np.transpose(chi_nu_p.dot(np.diag(1.0 / card["nu_p"])))

    # Calculate delayed and total fission multiplicities
    if max(card["fission"]) > 0.0:
        card["nu_f"][:] = card["nu_p"][:]
        for j in range(J):
            total = np.zeros(G)
            for i in range(N_nuclide):
                nuc = nuclides[i][0]
                density = nuclides[i][1]
                total += nuc["nu_d"][:, j] * nuc["fission"] * density
            card["nu_d"][:, j] = total / card["fission"]
            card["nu_f"] += card["nu_d"][:, j]

    # Push card
    mcdc.input_deck.materials.append(card)
    return card


def surface(type_, bc="interface", sensitivity=False, dsm_Np=1.0, **kw):
    """
    Create a surface card and return SurfaceHandle to define cell domain.

    Parameters
    ----------
    type_ : {'plane-x', 'plane-y', 'plane-z', 'plane', 'cylinder-x', 'cylinder-y',
             'cylinder-z', 'sphere', 'quadric'}
        Surface type.
    bc : {'interface', 'vacuum', 'reflective'}
        Surface boundary condition.
    sensitivity : bool, optional
        Set to `True` to calculate sensitivities to the nuclide
    dsm_Np : int
        Average number of derivative particles produced at each
        sensitivity surface crossing

    Other Parameters
    ----------------
    x : {float, array_like}
        x-position [cm] for `plane-x`. If a vector is passed, positions of the surface
        at the times specified by the parameter `t`.
    y : {float, array_like}
        y-position [cm] for `plane-y`. If a vector is passed, positions of the surface
        at the times specified by the parameter `t`.
    z : {float, array_like}
        z-position [cm] for `plane-z`. If a vector is passed, positions of the surface
        at the times specified by the parameter `t`.
    center : array_like
        Center point [cm] for `cylinder-x` (y,z), `cylinder-y` (x,z),
        `cylinder-z` (x,y), and `sphere` (x,y,z).
    radius : float
        Radius [cm] for `cylinder-x`, `cylinder-y`, `cylinder-z`, and `sphere`.
    A, B, C, D : float
        Coefficients [cm] for `plane`.
    A, B, C, D, E, F, G, H, I, J : float
        Coefficients [cm] for `quadric`.

    Returns
    -------
    SurfaceHandle
        A surface handle used for assigning surface, and its sense, to a cell card.

    See also
    --------
    mcdc.cell : SurfaceHandle is used to define cell domain
    """
    # Make surface card
    card = make_card_surface()
    card["ID"] = len(mcdc.input_deck.surfaces)

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
    # Set bc flags
    if bc == "vacuum":
        card["vacuum"] = True
    elif bc == "reflective":
        card["reflective"] = True

    # Sensitivity
    if sensitivity:
        # Set flag
        card["sensitivity"] = True
        mcdc.input_deck.technique["sensitivity"] = True
        mcdc.input_deck.technique["weighted_emission"] = False

        # Set ID
        mcdc.input_deck.setting["N_sensitivity"] += 1
        card["sensitivity_ID"] = mcdc.input_deck.setting["N_sensitivity"]

        # Set dsm_Np
        card["dsm_Np"] = dsm_Np

    # ==========================================================================
    # Surface attributes
    # ==========================================================================
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
    #   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0

    card["type"] = type_

    # Set up surface attributes
    if type_ == "plane-x":
        check_requirement("surface plane-x", kw, ["x"])
        card["G"] = 1.0
        card["linear"] = True
        if type(kw.get("x")) in [type([]), type(np.array([]))]:
            _set_J(kw.get("x"), kw.get("t"), card)
        else:
            card["J"][0, 0] = -kw.get("x")
    elif type_ == "plane-y":
        check_requirement("surface plane-y", kw, ["y"])
        card["H"] = 1.0
        card["linear"] = True
        if type(kw.get("y")) in [type([]), type(np.array([]))]:
            _set_J(kw.get("y"), kw.get("t"), card)
        else:
            card["J"][0, 0] = -kw.get("y")
    elif type_ == "plane-z":
        check_requirement("surface plane-z", kw, ["z"])
        card["I"] = 1.0
        card["linear"] = True
        if type(kw.get("z")) in [type([]), type(np.array([]))]:
            _set_J(kw.get("z"), kw.get("t"), card)
        else:
            card["J"][0, 0] = -kw.get("z")
    elif type_ == "plane":
        check_requirement("surface plane", kw, ["A", "B", "C", "D"])
        card["G"] = kw.get("A")
        card["H"] = kw.get("B")
        card["I"] = kw.get("C")
        card["J"][0, 0] = kw.get("D")
        card["linear"] = True
    elif type_ == "cylinder-x":
        check_requirement("surface cylinder-x", kw, ["center", "radius"])
        y, z = kw.get("center")[:]
        r = kw.get("radius")
        card["B"] = 1.0
        card["C"] = 1.0
        card["H"] = -2.0 * y
        card["I"] = -2.0 * z
        card["J"][0, 0] = y**2 + z**2 - r**2
    elif type_ == "cylinder-y":
        check_requirement("surface cylinder-y", kw, ["center", "radius"])
        x, z = kw.get("center")[:]
        r = kw.get("radius")
        card["A"] = 1.0
        card["C"] = 1.0
        card["G"] = -2.0 * x
        card["I"] = -2.0 * z
        card["J"][0, 0] = x**2 + z**2 - r**2
    elif type_ == "cylinder-z":
        check_requirement("surface cylinder-z", kw, ["center", "radius"])
        x, y = kw.get("center")[:]
        r = kw.get("radius")
        card["A"] = 1.0
        card["B"] = 1.0
        card["G"] = -2.0 * x
        card["H"] = -2.0 * y
        card["J"][0, 0] = x**2 + y**2 - r**2
    elif type_ == "sphere":
        check_requirement("surface sphere", kw, ["center", "radius"])
        x, y, z = kw.get("center")[:]
        r = kw.get("radius")
        card["A"] = 1.0
        card["B"] = 1.0
        card["C"] = 1.0
        card["G"] = -2.0 * x
        card["H"] = -2.0 * y
        card["I"] = -2.0 * z
        card["J"][0, 0] = x**2 + y**2 + z**2 - r**2
    elif type_ == "quadric":
        check_requirement(
            "surface quadric", kw, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        )
        card["A"] = kw.get("A")
        card["B"] = kw.get("B")
        card["C"] = kw.get("C")
        card["D"] = kw.get("D")
        card["E"] = kw.get("E")
        card["F"] = kw.get("F")
        card["G"] = kw.get("G")
        card["H"] = kw.get("H")
        card["I"] = kw.get("I")
        card["J"][0, 0] = kw.get("J")

    # Set normal vector if linear
    if card["linear"]:
        nx = card["G"]
        ny = card["H"]
        nz = card["I"]
        # Normalize
        norm = (nx**2 + ny**2 + nz**2) ** 0.5
        card["nx"] = nx / norm
        card["ny"] = ny / norm
        card["nz"] = nz / norm

    # Push card
    mcdc.input_deck.surfaces.append(card)
    return SurfaceHandle(card)


def _set_J(x, t, card):
    # Edit and add the edges
    t[0] = -SHIFT
    t = np.append(t, INF)
    x = np.append(x, x[-1])

    # Reset the constants
    card["J"] = np.zeros([0, 2])
    card["t"] = np.array([-SHIFT])

    # Iterate over inputs
    idx = 0
    for i in range(len(t) - 1):
        # Skip if step
        if t[i] == t[i + 1]:
            continue

        # Calculate constants
        J0 = x[i]
        J1 = (x[i + 1] - x[i]) / (t[i + 1] - t[i])

        # Append to card
        card["J"] = np.append(card["J"], [[J0, J1]], axis=0)
        card["t"] = np.append(card["t"], t[i + 1])

    card["J"] *= -1
    card["N_slice"] = len(card["J"])


def cell(surfaces_flags, fill, lattice_center=None):
    N_surface = len(surfaces_flags)

    # Make cell card
    card = make_card_cell(N_surface)
    card["ID"] = len(mcdc.input_deck.cells)

    # Surfaces and flags
    for i in range(N_surface):
        card["surface_IDs"][i] = surfaces_flags[i][0]["ID"]
        card["positive_flags"][i] = surfaces_flags[i][1]

    # Lattice cell?
    if fill["tag"] == "Lattice":
        card["lattice"] = True
        card["lattice_ID"] = fill["ID"]
        if lattice_center is not None:
            card["lattice_center"] = np.array(lattice_center)

    # Material cell
    else:
        card["material_ID"] = fill["ID"]
        card["material_name"] = fill["name"]

    # Push card
    mcdc.input_deck.cells.append(card)
    return card


def universe(cells, root=False):
    N_cell = len(cells)

    # Set default card values (c.f. type_.py)
    if not root:
        card = make_card_universe(N_cell)
        card["ID"] = len(mcdc.input_deck.universes)
    else:
        card = mcdc.input_deck.universes[0]
        card["N_cell"] = N_cell
        card["cell_IDs"] = np.zeros(N_cell, dtype=int)

    # Cells
    for i in range(N_cell):
        card["cell_IDs"][i] = cells[i]["ID"]

    # Push card
    if not root:
        mcdc.input_deck.universes.append(card)

    return card


def lattice(x=None, y=None, z=None, universes=None):
    # Make lattice card
    card = make_card_lattice()
    card["ID"] = len(mcdc.input_deck.lattices)

    # Set mesh
    if x is not None:
        card["mesh"]["x0"] = x[0]
        card["mesh"]["dx"] = x[1]
        card["mesh"]["Nx"] = x[2]
    if y is not None:
        card["mesh"]["y0"] = y[0]
        card["mesh"]["dy"] = y[1]
        card["mesh"]["Ny"] = y[2]
    if z is not None:
        card["mesh"]["z0"] = z[0]
        card["mesh"]["dz"] = z[1]
        card["mesh"]["Nz"] = z[2]

    # Set universe IDs
    universe_IDs = np.array(universes, dtype=np.int64)
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
    card["universe_IDs"] = np.flip(tmp, axis=2)

    # Push card
    mcdc.input_deck.lattices.append(card)
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
        Flag for isotropic source
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
        A source card
    """
    # Check the suplied keyword arguments
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
    card = make_card_source()
    card["ID"] = len(mcdc.input_deck.sources)

    # Set position
    if point is not None:
        card["x"] = point[0]
        card["y"] = point[1]
        card["z"] = point[2]
    else:
        card["box"] = True
        if x is not None:
            card["box_x"] = np.array(x)
        if y is not None:
            card["box_y"] = np.array(y)
        if z is not None:
            card["box_z"] = np.array(z)

    # Set direction
    if white is not None:
        card["isotropic"] = False
        card["white"] = True
        ux = white[0]
        uy = white[1]
        uz = white[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card["white_x"] = ux / norm
        card["white_y"] = uy / norm
        card["white_z"] = uz / norm
    elif direction is not None:
        card["isotropic"] = False
        ux = direction[0]
        uy = direction[1]
        uz = direction[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card["ux"] = ux / norm
        card["uy"] = uy / norm
        card["uz"] = uz / norm

    # Set energy
    if energy is not None:
        if mcdc.input_deck.setting["mode_MG"]:
            group = np.array(energy)
            # Normalize
            card["group"] = group / np.sum(group)
        if mcdc.input_deck.setting["mode_CE"]:
            energy = np.array(energy)
            # Resize
            card["energy"] = np.zeros(energy.shape)
            # Set energy
            card["energy"][0, :] = energy[0, :]
            # Normalize pdf
            card["energy"][1, :] = energy[1, :] / np.trapz(energy[1, :], x=energy[0, :])
            # Make cdf
            card["energy"][1, :] = sp.integrate.cumulative_trapezoid(
                card["energy"][1], x=card["energy"][0], initial=0.0
            )

    # Set time
    if time is not None:
        card["time"] = np.array(time)

    # Set probability
    if prob is not None:
        card["prob"] = prob

    # Push card
    mcdc.input_deck.sources.append(card)
    return card


def tally(
    scores,
    x=np.array([-INF, INF]),
    y=np.array([-INF, INF]),
    z=np.array([-INF, INF]),
    t=np.array([-INF, INF]),
    mu=np.array([-1.0, 1.0]),
    azi=np.array([-PI, PI]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
):
    # Get tally card
    card = mcdc.input_deck.tally

    # Set mesh
    card["mesh"]["x"] = x
    card["mesh"]["y"] = y
    card["mesh"]["z"] = z
    card["mesh"]["t"] = t
    card["mesh"]["mu"] = mu
    card["mesh"]["azi"] = azi

    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = mcdc.input_deck.materials[0]["G"]
        card["mesh"]["g"] = np.linspace(0, G, G + 1) - 0.5
    else:
        card["mesh"]["g"] = g
    if mcdc.input_deck.setting["mode_CE"]:
        card["mesh"]["g"] = E

    # Set score flags
    for s in scores:
        found = False
        for score_name in type_.score_list:
            if s.replace("-", "_") == score_name:
                card["tracklength"] = True
                card[score_name] = True
                found = True
                break
        if not found:
            print_error("Unknown tally score %s" % s)

    return card


# ==============================================================================
# Setting
# ==============================================================================


def setting(**kw):
    # Check the suplied keyword arguments
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

    # Check if setting card has been initialized
    card = mcdc.input_deck.setting

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
        card_setting = mcdc.input_deck.setting
        with h5py.File(source_file, "r") as f:
            card_setting["N_particle"] = f["particles_size"][()]

    # IC file
    if IC_file is not None:
        card["IC_file"] = True
        card["IC_file_name"] = IC_file

        # Set number of particles
        card_setting = mcdc.input_deck.setting
        with h5py.File(IC_file, "r") as f:
            card_setting["N_particle"] = f["IC/neutrons_size"][()]
            card_setting["N_precursor"] = f["IC/precursors_size"][()]

    # TODO: Allow both source and IC files
    if IC_file and source_file:
        print_error("Using both source and IC files is not supported yet.")


def eigenmode(
    N_inactive=0, N_active=0, k_init=1.0, gyration_radius=None, save_particle=False
):
    # Update setting card
    card = mcdc.input_deck.setting
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

    # Update tally card
    card = mcdc.input_deck.tally
    card["tracklength"] = True


# ==============================================================================
# Technique
# ==============================================================================


def implicit_capture():
    card = mcdc.input_deck.technique
    card["implicit_capture"] = True
    card["weighted_emission"] = False


def weighted_emission(flag):
    card = mcdc.input_deck.technique
    card["weighted_emission"] = flag


def population_control(pct="combing"):
    card = mcdc.input_deck.technique
    card["population_control"] = True
    card["weighted_emission"] = False
    if pct == "combing":
        card["pct"] = PCT_COMBING
    elif pct == "combing-weight":
        card["pct"] = PCT_COMBING_WEIGHT
    else:
        print_error("Unknown PCT type " + pct)


def branchless_collision():
    card = mcdc.input_deck.technique
    card["branchless_collision"] = True
    card["weighted_emission"] = False


def time_census(t):
    # Remove census beyond the final tally time grid point
    while True:
        if t[-1] >= mcdc.input_deck.tally["mesh"]["t"][-1]:
            t = t[:-1]
        else:
            break

    # Add the default, final census-at-infinity
    t = np.append(t, INF)

    # Set the time census parameters
    card = mcdc.input_deck.setting
    card["census_time"] = t
    card["N_census"] = len(t)


def weight_window(x=None, y=None, z=None, t=None, window=None, width=None):
    card = mcdc.input_deck.technique
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
    source_xy0=None,
    source_xz0=None,
    source_yz0=None,
    source_xyz0=None,
    fission_source0=None,
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
    card = mcdc.input_deck.technique
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

    if eigenmode_solver == "davidson":
        card["iqmc"]["krylov_vector_size"] += 1

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

    if score_list["tilt-xy"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_xy0 is None:
            source_xy0 = np.zeros_like(phi0)

    if score_list["tilt-xz"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_xz0 is None:
            source_xz0 = np.zeros_like(phi0)

    if score_list["tilt-yz"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_yz0 is None:
            source_yz0 = np.zeros_like(phi0)

    if fission_source0 is not None:
        card["iqmc"]["score"]["fission-source"] = fission_source0

    card["iqmc"]["score"]["flux"] = phi0
    card["iqmc"]["score"]["tilt-x"] = source_x0
    card["iqmc"]["score"]["tilt-y"] = source_y0
    card["iqmc"]["score"]["tilt-z"] = source_z0
    card["iqmc"]["score"]["tilt-xy"] = source_xy0
    card["iqmc"]["score"]["tilt-xz"] = source_xz0
    card["iqmc"]["score"]["tilt-yz"] = source_yz0
    card["iqmc"]["source"] = source0
    card["iqmc"]["fixed_source"] = fixed_source
    card["iqmc"]["fixed_source_solver"] = fixed_source_solver
    card["iqmc"]["eigenmode_solver"] = eigenmode_solver
    card["iqmc"]["preconditioner_sweeps"] = preconditioner_sweeps
    card["iqmc"]["krylov_restart"] = krylov_restart


def weight_roulette(chance, wr_threshold):
    """
    If neutron weight is below wr_threshold, then enter weight rouelette
    technique. Neutron has 'chance' probability of having its weight increased
    by factor of 1/CHANCE, and 1-CHANCE probability of terminating.

    Parameters
    ----------
    chance : probability of survival
    wr_threshold : weight_roulette() is called on a particle
                    if P['w'] <= wr_threshold

    Returns
    -------
    None.

    """
    card = mcdc.input_deck.technique
    card["weight_roulette"] = True
    card["wr_chance"] = chance
    card["wr_threshold"] = wr_threshold


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
    Turn on initial condition generator, which samples initial neutrons and precursors
    during an eigenvalue simulation.


    Parameters
    ----------
    N_neutron : int
        Neutron target size
    N_precursor : int
        Delayed neutron precursot target size
    cycle_stretch : float
        Factor to strethch number of cycles. Higher cycle stretch reduces inter-cycle
        correlation.
    neutron_density, max_neutron_density : float
        Total and maximum neutron density, required if `N_neutron` > 0.
    precursor_density, max_precursor_density : float
        Total and maximum precursor density, required if `N_precursor` > 0.
    """

    # Turn on eigenmode and population control
    eigenmode()
    population_control()

    # Set parameters
    card = mcdc.input_deck.technique
    card["IC_generator"] = True
    card["IC_N_neutron"] = N_neutron
    card["IC_N_precursor"] = N_precursor

    # Setting parameters
    card_setting = mcdc.input_deck.setting
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
    card = mcdc.input_deck.technique
    if order > 2:
        print_error("DSM currently only supports up to second-order sensitivities")
    card["dsm_order"] = order


def uq(**kw):
    def append_card(delta_card, global_tag):
        delta_card["distribution"] = dist
        delta_card["flags"] = []
        for key in kw.keys():
            check_support(parameter["tag"] + " parameter", key, parameter_list, False)
            delta_card["flags"].append(key)
            delta_card[key] = kw[key]
        mcdc.input_deck.uq_deltas[global_tag].append(delta_card)

    mcdc.input_deck.technique["uq"] = True
    # Make sure N_batch > 1
    if mcdc.input_deck.setting["N_batch"] <= 1:
        print_error("Must set N_batch>1 with mcdc.setting() prior to mcdc.uq() call.")

    # Check uq parameter
    parameter_ = check_support(
        "uq parameter",
        list(kw)[0],
        ["nuclide", "material", "surface", "source"],
        False,
    )
    parameter = kw[parameter_]
    del kw[parameter_]
    parameter["uq"] = True

    # Confirm supplied distribution
    check_requirement("uq", kw, ["distribution"])
    dist = check_support("distribution", kw["distribution"], ["uniform"], False)
    del kw["distribution"]

    # Only remaining keywords should be the parameter delta(s)

    if parameter["tag"] == "Material":
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
        if parameter["N_nuclide"] == 1:
            nuc_card = make_card_nuclide(parameter["G"], parameter["J"])
            nuc_card["ID"] = parameter["nuclide_IDs"][0]
            append_card(nuc_card, "nuclides")
        delta_card = make_card_material(
            parameter["N_nuclide"], parameter["G"], parameter["J"]
        )
        for name in ["ID", "nuclide_IDs", "nuclide_densities"]:
            delta_card[name] = parameter[name]
    elif parameter["tag"] == "Nuclide":
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
        delta_card = make_card_nuclide(parameter["G"], parameter["J"])
        delta_card["ID"] = parameter["ID"]
    append_card(delta_card, global_tag)
    # elif parameter['tag'] is 'Surface':
    # elif parameter['tag'] is 'Source':


# ==============================================================================
# Util
# ==============================================================================


def nuclide_registered(name):
    for card in mcdc.input_deck.nuclides:
        if name == card["name"]:
            return True
    return False


def get_nuclide(name):
    for card in mcdc.input_deck.nuclides:
        if name == card["name"]:
            return card


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


def reset_cards():
    mcdc.input_deck.reset()
