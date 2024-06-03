import numpy as np

from mcdc.constant import INF, GYRATION_RADIUS_ALL, PCT_NONE, PI


# ======================================================================================
# Input Deck
# ======================================================================================


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


class InputDeck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nuclides = []
        self.materials = []
        self.surfaces = []
        self.regions = []
        self.cells = []
        self.universes = [None]  # Placeholder for the root universe
        self.lattices = []
        self.sources = []

        self.tally = {
            "tag": "Tally",
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
                "fixed_source": np.ones([1, 1, 1, 1, 1]),
                "material_idx": np.ones([1, 1, 1, 1]),
                "source": np.ones([1, 1, 1, 1, 1]),
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


input_deck = InputDeck()
