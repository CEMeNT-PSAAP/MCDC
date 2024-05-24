import numpy as np

from mcdc.constant import INF, GYRATION_RADIUS_ALL, PCT_NONE, PI, SHIFT


class InputDeck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nuclides = []
        self.materials = []
        self.surfaces = []
        self.cells = []
        self.universes = [{}]
        self.lattices = []
        self.sources = []
        self.meshes = []
        # Default cards are set by functions make_card_*

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
                    "fission-source": np.zeros([1, 1, 1, 1]),
                },
                "score_list": {
                    "flux": True,
                    "effective-scattering": True,
                    "effective-fission": True,
                    "tilt-x": False,
                    "tilt-y": False,
                    "tilt-z": False,
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


class SurfaceHandle:
    def __init__(self, card):
        self.card = card

    def __pos__(self):
        return [self.card, True]

    def __neg__(self):
        return [self.card, False]


def make_card_nuclide(G=1, J=0):
    card = {}
    card["tag"] = "Nuclide"
    card["name"] = ""
    card["ID"] = -1
    card["fissionable"] = False
    card["G"] = G
    card["J"] = J
    card["speed"] = np.ones(G)
    card["decay"] = np.ones(J) * INF
    card["capture"] = np.zeros(G)
    card["scatter"] = np.zeros(G)
    card["fission"] = np.zeros(G)
    card["total"] = np.zeros(G)
    card["nu_s"] = np.ones(G)
    card["nu_p"] = np.zeros(G)
    card["nu_d"] = np.zeros([G, J])
    card["nu_f"] = np.zeros(G)
    card["chi_s"] = np.zeros([G, G])
    card["chi_p"] = np.zeros([G, G])
    card["chi_d"] = np.zeros([J, G])
    card["sensitivity"] = False
    card["sensitivity_ID"] = 0
    card["dsm_Np"] = 1.0
    card["uq"] = False
    return card


def make_card_material(N_nuclide, G=1, J=0):
    card = {}
    card["tag"] = "Material"
    card["ID"] = -1
    card["N_nuclide"] = N_nuclide
    card["nuclide_IDs"] = np.zeros(N_nuclide, dtype=int)
    card["nuclide_densities"] = np.zeros(N_nuclide, dtype=float)
    card["G"] = G
    card["J"] = J
    card["speed"] = np.zeros(G)
    card["capture"] = np.zeros(G)
    card["scatter"] = np.zeros(G)
    card["fission"] = np.zeros(G)
    card["total"] = np.zeros(G)
    card["nu_s"] = np.ones(G)
    card["nu_p"] = np.zeros(G)
    card["nu_d"] = np.zeros([G, J])
    card["nu_f"] = np.zeros(G)
    card["chi_s"] = np.zeros([G, G])
    card["chi_p"] = np.zeros([G, G])
    card["name"] = None
    card["sensitivity"] = False
    card["uq"] = False
    return card


def make_card_surface():
    card = {}
    card["tag"] = "Surface"
    card["ID"] = -1
    card["vacuum"] = False
    card["reflective"] = False
    card["A"] = 0.0
    card["B"] = 0.0
    card["C"] = 0.0
    card["D"] = 0.0
    card["E"] = 0.0
    card["F"] = 0.0
    card["G"] = 0.0
    card["H"] = 0.0
    card["I"] = 0.0
    card["J"] = np.array([[0.0, 0.0]])
    card["t"] = np.array([-SHIFT, INF])
    card["N_slice"] = 1
    card["linear"] = False
    card["nx"] = 0.0
    card["ny"] = 0.0
    card["nz"] = 0.0
    card["sensitivity"] = False
    card["sensitivity_ID"] = 0
    card["type"] = " "
    card["dsm_Np"] = 1.0
    return card


def make_card_cell(N_surface):
    card = {}
    card["tag"] = "Cell"
    card["ID"] = -1
    card["N_surface"] = N_surface
    card["surface_IDs"] = np.zeros(N_surface, dtype=int)
    card["positive_flags"] = np.zeros(N_surface, dtype=bool)
    card["material_ID"] = 0
    card["material_name"] = None
    card["lattice"] = False
    card["lattice_ID"] = 0
    card["lattice_center"] = np.array([0.0, 0.0, 0.0])
    return card


def make_card_universe(N_cell):
    card = {}
    card["tag"] = "Universe"
    card["ID"] = -1
    card["N_cell"] = N_cell
    card["cell_IDs"] = np.zeros(N_cell, dtype=int)
    return card


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
