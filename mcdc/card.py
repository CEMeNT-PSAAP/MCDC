import numpy as np

from mcdc.constant import INF, SHIFT, PI

# Get the global variable container
import mcdc.global_ as global_


class InputCard:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        text = "%s card\n" % self.tag

        for name in [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a)) and a != "tag"
        ]:
            text += "  %s : %s\n" % (name, str(getattr(self, name)))
        return text


class NuclideCard(InputCard):
    def __init__(self, G=1, J=0):
        InputCard.__init__(self, "Nuclide")

        # Set card data
        self.ID = None
        self.G = G
        self.J = J
        self.fissionable = False
        self.speed = np.ones(G)
        self.decay = np.ones(J) * INF
        self.capture = np.zeros(G)
        self.scatter = np.zeros(G)
        self.fission = np.zeros(G)
        self.total = np.zeros(G)
        self.nu_s = np.ones(G)
        self.nu_p = np.zeros(G)
        self.nu_d = np.zeros([G, J])
        self.nu_f = np.zeros(G)
        self.chi_s = np.zeros([G, G])
        self.chi_p = np.zeros([G, G])
        self.chi_d = np.zeros([J, G])
        self.sensitivity = False
        self.sensitivity_ID = 0
        self.dsm_Np = 1.0
        self.uq = False
        self.flags = []
        self.distribution = ""


class MaterialCard(InputCard):
    def __init__(self, N_nuclide, G=1, J=0):
        InputCard.__init__(self, "Material")

        # Set card data
        self.ID = None
        self.N_nuclide = N_nuclide
        self.nuclide_IDs = np.zeros(N_nuclide, dtype=int)
        self.nuclide_densities = np.zeros(N_nuclide, dtype=float)
        self.G = G
        self.J = J
        self.speed = np.zeros(G)
        self.capture = np.zeros(G)
        self.scatter = np.zeros(G)
        self.fission = np.zeros(G)
        self.total = np.zeros(G)
        self.nu_s = np.ones(G)
        self.nu_p = np.zeros(G)
        self.nu_d = np.zeros([G, J])
        self.nu_f = np.zeros(G)
        self.chi_s = np.zeros([G, G])
        self.chi_p = np.zeros([G, G])
        self.sensitivity = False
        self.uq = False
        self.flags = []
        self.distribution = ""


class RegionCard(InputCard):
    def __init__(self, type_):
        InputCard.__init__(self, "Region")

        # Set card data
        self.ID = None
        self.type = type_
        self.A = -1
        self.B = -1

    def __and__(self, other):
        region = RegionCard("intersection")
        region.A = self.ID
        region.B = other.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __or__(self, other):
        region = RegionCard("union")
        region.A = self.ID
        region.B = other.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __invert__(self):
        region = RegionCard("complement")
        region.A = self.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region


class SurfaceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Surface")

        # Set card data
        self.ID = None
        self.type = ""
        self.boundary_type = "interface"
        self.linear = False
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.E = 0.0
        self.F = 0.0
        self.G = 0.0
        self.H = 0.0
        self.I = 0.0
        self.J = np.array([[0.0, 0.0]])
        self.t = np.array([-SHIFT, INF])
        self.N_slice = 1
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0
        self.sensitivity = False
        self.sensitivity_ID = 0
        self.dsm_Np = 1.0
        self.N_tally = 0
        self.tally_IDs = []

    def __pos__(self):
        region = RegionCard("halfspace")
        region.A = self.ID
        region.B = 1
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __neg__(self):
        region = RegionCard("halfspace")
        region.A = self.ID
        region.B = 0
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region


class CellCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.ID = None
        self.region_ID = -1
        self.fill_type = "material"
        self.fill_ID = -1
        self.translation = np.array([0.0, 0.0, 0.0])
        self.N_surface = 0
        self.surface_ID = np.zeros(0, dtype=int)


class UniverseCard(InputCard):
    def __init__(self, N_cell):
        InputCard.__init__(self, "Universe")

        # Set card data
        self.ID = None
        self.N_cell = N_cell
        self.cell_IDs = np.zeros(N_cell, dtype=int)


class LatticeCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Lattice")

        # Set card data
        self.ID = None
        self.universe_IDs = np.array([[[[0]]]])
        self.mesh = {
            "x0": -INF,
            "dx": 2 * INF,
            "Nx": 1,
            "y0": -INF,
            "dy": 2 * INF,
            "Ny": 1,
            "z0": -INF,
            "dz": 2 * INF,
            "Nz": 1,
        }


class SourceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Source")

        # Set card data
        self.ID = None
        self.box = False
        self.isotropic = True
        self.white = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.box_x = np.array([0.0, 0.0])
        self.box_y = np.array([0.0, 0.0])
        self.box_z = np.array([0.0, 0.0])
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.white_x = 0.0
        self.white_y = 0.0
        self.white_z = 0.0
        self.group = np.array([1.0])
        self.energy = np.array([[14e6, 14e6], [1.0, 1.0]])
        self.time = np.array([0.0, 0.0])
        self.prob = 1.0


# ======================================================================================
# Tally cards
# ======================================================================================


class TallyCard(InputCard):
    def __init__(self, type_):
        InputCard.__init__(self, type_)

        # Set card data
        self.ID = None
        self.scores = []
        self.N_bin = 0

        # Filters
        self.t = np.array([-INF, INF])
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.g = np.array([-INF, INF])


class MeshTallyCard(TallyCard):
    def __init__(self):
        TallyCard.__init__(self, "Mesh tally")

        # Set card data
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.N_bin = 1


class SurfaceTallyCard(TallyCard):
    def __init__(self, surface_ID):
        TallyCard.__init__(self, "Surface tally")

        # Set card data
        self.surface_ID = surface_ID
        self.N_bin = 1
