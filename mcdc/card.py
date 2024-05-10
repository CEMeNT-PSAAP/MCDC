import numpy as np

from mcdc.constant import INF, GYRATION_RADIUS_ALL, PCT_NONE, PI, SHIFT


class InputDeck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nuclides = []
        self.materials = []
        self.surfaces = []
        self.regions = []
        self.cells = []
        self.universes = [{}]
        self.lattices = []
        self.sources = []
        self.meshes = []

        # Root universe
        self.universes[0] = make_card_universe(0)
        self.universes[0]["ID"] = 0

        self.tally = {
            "tag": "Tally",
            "tracklength": True,
            "flux": False,
            "density": False,
            "fission": False,
            "total": False,
            "current": False,
            "eddington": False,
            "exit": False,
            "mesh": make_card_mesh(),
        }

        self.setting = {
            "tag": "Setting",
            "mode_MG": True,
            "mode_CE": False,
            "N_particle": 0,
            "N_batch": 1,
            "rng_seed": 1,
            "time_boundary": INF,
            "progress_bar": True,
            "output_name": "output",
            "save_input_deck": True,
            "track_particle": False,
            "mode_eigenvalue": False,
            "k_init": 1.0,
            "N_inactive": 0,
            "N_active": 0,
            "N_cycle": 0,
            "caching": True,
            "save_particle": False,
            "gyration_radius": False,
            "gyration_radius_type": GYRATION_RADIUS_ALL,
            "N_census": 1,
            "census_time": np.array([INF]),
            "source_file": False,
            "source_file_name": "",
            "IC_file": False,
            "IC_file_name": "",
            "N_precursor": 0,
            # Below are parameters not copied to mcdc.setting
            "bank_active_buff": 100,
            "bank_census_buff": 1.0,
            # TODO: Move to technique
            "N_sensitivity": 0,
        }

        self.technique = {
            "tag": "Technique",
            "weighted_emission": True,
            "implicit_capture": False,
            "population_control": False,
            "pct": PCT_NONE,
            "pc_factor": 1.0,
            "weight_window": False,
            "ww": np.ones([1, 1, 1, 1]),
            "ww_width": 2.5,
            "ww_mesh": make_card_mesh(),
            "domain_decomposition": False,
            "dd_idx": 0,
            "dd_mesh": make_card_mesh(),
            "dd_exchange_rate": 0,
            "dd_repro": False,
            "dd_work_ratio": np.array([1]),
            "weight_roulette": False,
            "wr_threshold": 0.0,
            "wr_survive": 1.0,
            "iQMC": False,
            "iqmc": {
                "generator": "sobol",
                "fixed_source_solver": "source_iteration",
                "eigenmode_solver": "power_iteration",
                "preconditioner_sweeps": 5,
                "krylov_restart": 5,
                "krylov_vector_size": 1,
                "tol": 1e-6,
                "res": 1.0,
                "res_outter": 1.0,
                "itt": 0,
                "itt_outter": 0,
                "maxitt": 5,
                "N_dim": 6,
                "seed": 12345,
                "scramble": False,
                "fixed_source": np.ones([1, 1, 1, 1]),
                "material_idx": np.ones([1, 1, 1, 1]),
                "source": np.ones([1, 1, 1, 1]),
                "score": {
                    "flux": np.ones([1, 1, 1, 1]),
                    "tilt-x": np.zeros([1, 1, 1, 1]),
                    "tilt-y": np.zeros([1, 1, 1, 1]),
                    "tilt-z": np.zeros([1, 1, 1, 1]),
                    "tilt-xy": np.zeros([1, 1, 1, 1]),
                    "tilt-xz": np.zeros([1, 1, 1, 1]),
                    "tilt-yz": np.zeros([1, 1, 1, 1]),
                    "fission-source": np.zeros([1, 1, 1, 1]),
                },
                "score_list": {
                    "flux": True,
                    "effective-scattering": True,
                    "effective-fission": True,
                    "tilt-x": False,
                    "tilt-y": False,
                    "tilt-z": False,
                    "tilt-xy": False,
                    "tilt-xz": False,
                    "tilt-yz": False,
                    "fission-power": False,
                    "fission-source": False,
                },
                "mesh": {
                    "g": np.array([-INF, INF]),
                    "t": np.array([-INF, INF]),
                    "x": np.array([-INF, INF]),
                    "y": np.array([-INF, INF]),
                    "z": np.array([-INF, INF]),
                    "mu": np.array([-1.0, 1.0]),
                    "azi": np.array([-PI, PI]),
                },
            },
            "IC_generator": False,
            "IC_N_neutron": 0,
            "IC_N_precursor": 0,
            "IC_neutron_density": 0.0,
            "IC_precursor_density": 0.0,
            "IC_neutron_density_max": 0.0,
            "IC_precursor_density_max": 0.0,
            "IC_cycle_stretch": 1.0,
            "branchless_collision": False,
            "dsm_order": 1,
            "uq": False,
        }

        self.uq_deltas = {
            "tag": "Uq",
            "nuclides": [],
            "materials": [],
            "surfaces": [],
        }


# =============================================================================
# Cards
# =============================================================================


class InputCard:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        print(self.tag + " card")
        for key in self.data:
            print("  " + key + " : " + str(self.data[key]))


class NuclideCard(InputCard):
    count = 0
    def __init__(self, G=1, J=0):
        InputCard.__init__(self, "Nuclide")

        # Set card data
        self.data = {
            "name": "Nuclide %i" % NuclideCard.count,
            "ID": NuclideCard.count,
            "fissionable": False,
            "G": G,
            "J": J,
            "speed": np.ones(G),
            "decay": np.ones(J) * INF,
            "capture": np.zeros(G),
            "scatter": np.zeros(G),
            "fission": np.zeros(G),
            "total": np.zeros(G),
            "nu_s": np.ones(G),
            "nu_p": np.zeros(G),
            "nu_d": np.zeros([G, J]),
            "nu_f": np.zeros(G),
            "chi_s": np.zeros([G, G]),
            "chi_p": np.zeros([G, G]),
            "chi_d": np.zeros([J, G]),
            "sensitivity": False,
            "sensitivity_ID": 0,
            "dsm_Np": 1.0,
            "uq": False,
        }

        NuclideCard.count += 1


class MaterialCard(InputCard):
    count = 0

    def __init__(self, N_nuclide, G=1, J=0):
        InputCard.__init__(self, "Material")

        # Set card data
        self.data = {
            "name": "Material %i" % MaterialCard.count
            "ID": MaterialCard.count
            "N_nuclide": N_nuclide
            "nuclide_IDs": np.zeros(N_nuclide, dtype=int)
            "nuclide_densities": np.zeros(N_nuclide, dtype=float)
            "G": G
            "J": J
            "speed": np.zeros(G)
            "capture": np.zeros(G)
            "scatter": np.zeros(G)
            "fission": np.zeros(G)
            "total": np.zeros(G)
            "nu_s": np.ones(G)
            "nu_p": np.zeros(G)
            "nu_d": np.zeros([G, J])
            "nu_f": np.zeros(G)
            "chi_s": np.zeros([G, G])
            "chi_p": np.zeros([G, G])
            "sensitivity": False
            "uq": False
        }

        MaterialCard.count += 1

class RegionCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Region")

        # Set card data
        self.data = {
            "ID": RegionCard.count,
            'half_space': False,
            "intersection": False,
            "union": False,
            "complement": False,
            "A": -1,
            "B": -1,
        }

        RegionCard.count += 1

    def __AND__(self, other):
        region = RegionCard()
        region.data['intersection'] = True
        region.data['A'] = self.data['ID']
        region.data['B'] = other.data['ID']
        return region

    def __OR__(self, other):
        region = RegionCard()
        region.data['union'] = True
        region.data['A'] = self.data['ID']
        region.data['B'] = other.data['ID']
        return region

    def __INVERT__(self, other):
        region = RegionCard()
        region.data['complement'] = True
        region.data['A'] = self.data['ID']
        return region


class SurfaceCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Surface")

        # Set card data
        self.data = {
            "name": "Surface %i" % SurfaceCard.count,
            "ID": SurfaceCard.count,
            "type": "",
            "vacuum": False,
            "reflective": False,
            "linear": False,
            "A": 0.0,
            "B": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "G": 0.0,
            "H": 0.0,
            "I": 0.0,
            "J": np.array([[0.0, 0.0]]),
            "t": np.array([-SHIFT, INF]),
            "N_slice": 1,
            "nx": 0.0,
            "ny": 0.0,
            "nz": 0.0,
            "sensitivity": False,
            "sensitivity_ID": 0,
            "dsm_Np": 1.0,
        }

        SurfaceCard.count += 1

    def __pos__(self):
        region = RegionCard()
        region.data['half_space'] = True
        region.data['A'] = self.data['ID']
        region.data['B'] = 1
        return region

    def __neg__(self):
        region = RegionCard()
        region.data['half_space'] = True
        region.data['A'] = self.data['ID']
        region.data['B'] = 0
        return region


class CellCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.data = {
            "name": "Cell %i" % CellCard.count,
            "ID": CellCard.count,
            "region_ID": 0
            "material_ID": 0
            "lattice": False
            "lattice_ID": 0
            "lattice_center": np.array([0.0, 0.0, 0.0])
        }

        CellCard.count += 1


class UniverseCard(InputCard):
    count = 0

    def __init__(self, N)cell):
        InputCard.__init__(self, "Universe")

        # Set card data
        self.data = {
            "name": "Universe %i" % UniverseCard.count,
            "ID": universeCard.count,
            "N_cell": N_cell
            "cell_IDs": np.zeros(N_cell, dtype=int)
        }

        UniverseCard.count += 1


def make_card_lattice():
    card = {}
    card["tag"] = "Lattice"
    card["ID"] = -1
    card["mesh"] = {
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
    card["universe_IDs"] = np.array([[[[0]]]])
    return card


class LatticeCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Lattice")

        # Set card data
        self.data = {
            "name": "Cell %i" % LatticeCard.count,
            "ID": LatticeCard.count,
            "universe_IDs": np.array([[[[0]]]]),
            "mesh": {
                "x0": -INF,
                "dx": 2 * INF,
                "Nx": 1,
                "y0": -INF,
                "dy": 2 * INF,
                "Ny": 1,
                "z0": -INF,
                "dz": 2 * INF,
                "Nz": 1,
            },
        }

        LatticeCard.count += 1


def make_card_source():
    card = {}
    card["tag"] = "Source"
    card["ID"] = -1
    card["box"] = False
    card["isotropic"] = True
    card["white"] = False
    card["x"] = 0.0
    card["y"] = 0.0
    card["z"] = 0.0
    card["box_x"] = np.array([0.0, 0.0])
    card["box_y"] = np.array([0.0, 0.0])
    card["box_z"] = np.array([0.0, 0.0])
    card["ux"] = 0.0
    card["uy"] = 0.0
    card["uz"] = 0.0
    card["white_x"] = 0.0
    card["white_y"] = 0.0
    card["white_z"] = 0.0
    card["group"] = np.array([1.0])
    card["energy"] = np.array([[14e6, 14e6], [1.0, 1.0]])
    card["time"] = np.array([0.0, 0.0])
    card["prob"] = 1.0
    return card


class CellCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.data = {
            "name": "Cell %i" % CellCard.count,
            "ID": CellCard.count,
            "region_ID": 0
            "material_ID": 0
            "lattice": False
            "lattice_ID": 0
            "lattice_center": np.array([0.0, 0.0, 0.0])
        }

        CellCard.count += 1


def make_card_mesh():
    return {
        "x": np.array([-INF, INF]),
        "y": np.array([-INF, INF]),
        "z": np.array([-INF, INF]),
        "t": np.array([-INF, INF]),
        "mu": np.array([-1.0, 1.0]),
        "azi": np.array([-PI, PI]),
        "g": np.array([-INF, INF]),
    }


class CellCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.data = {
            "name": "Cell %i" % CellCard.count,
            "ID": CellCard.count,
            "region_ID": 0
            "material_ID": 0
            "lattice": False
            "lattice_ID": 0
            "lattice_center": np.array([0.0, 0.0, 0.0])
        }

        CellCard.count += 1


def make_card_uq():
    return {
        "tag": "t",
        "ID": -1,
        "key": "k",
        "mean": 0.0,
        "delta": 0.0,
        "distribution": "d",
        "rng_seed": 0,
        "group": False,
        "group_group": False,
    }
class CellCard(InputCard):
    count = 0

    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.data = {
            "name": "Cell %i" % CellCard.count,
            "ID": CellCard.count,
            "region_ID": 0
            "material_ID": 0
            "lattice": False
            "lattice_ID": 0
            "lattice_center": np.array([0.0, 0.0, 0.0])
        }

        CellCard.count += 1


