import h5py
import numpy as np
from scipy.stats import qmc

from mpi4py import MPI

import mcdc.kernel as kernel
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.loop import loop_main, loop_iqmc
from mcdc.print_ import print_banner, print_msg, print_runtime, print_header_eigenvalue

# Get input_card and set global variables as "mcdc"
import mcdc.global_ as mcdc_

input_card = mcdc_.input_card
mcdc = mcdc_.global_

kernel.adapt_utils('cpu')

def run():
    # Start timer
    total_start = MPI.Wtime()

    # Preparation:
    #   process input cards, make types, and allocate global variables
    preparation_start = MPI.Wtime()
    prepare()
    mcdc["runtime_preparation"] = MPI.Wtime() - preparation_start

    # Print banner, hardware configuration, and header
    print_banner(mcdc)
    print_msg(" Now running TNT...")
    if mcdc["setting"]["mode_eigenvalue"]:
        print_header_eigenvalue(mcdc)

    # Run simulation
    simulation_start = MPI.Wtime()
    if not mcdc["technique"]["iQMC"]:
        loop_main(mcdc)
    else:
        loop_iqmc(mcdc)
    mcdc["runtime_simulation"] = MPI.Wtime() - simulation_start

    # Output: generate hdf5 output files
    output_start = MPI.Wtime()
    generate_hdf5()
    mcdc["runtime_output"] = MPI.Wtime() - output_start

    # Stop timer
    MPI.COMM_WORLD.Barrier()
    mcdc["runtime_total"] = MPI.Wtime() - total_start

    # Closout
    closeout()


def prepare():
    global mcdc

    # =========================================================================
    # Sizes
    #   We need this to determine the maximum size of model objects
    # =========================================================================

    # Neutron and delayed neutron precursor group sizes
    #   We assume that all materials have the same group structures
    G = input_card.materials[0]["G"]
    J = input_card.materials[0]["J"]

    # Number of model objects
    N_nuclide = len(input_card.nuclides)
    N_material = len(input_card.materials)
    N_surface = len(input_card.surfaces)
    N_cell = len(input_card.cells)
    N_universe = len(input_card.universes)
    N_lattice = len(input_card.lattices)
    N_source = len(input_card.sources)

    # Simulation parameters
    N_particle = input_card.setting["N_particle"]
    N_cycle = input_card.setting["N_cycle"]

    # Maximum object references
    # Maximum nuclides per material
    Nmax_nuclide = max([material["N_nuclide"] for material in input_card.materials])
    # Maximum surfaces per cell
    Nmax_surface = max([cell["N_surface"] for cell in input_card.cells])
    # Maximum cells per universe
    Nmax_cell = max([universe["N_cell"] for universe in input_card.universes])

    # Maximum time-dependent surface slices
    Nmax_slice = 0
    for surface in input_card.surfaces:
        Nmax_slice = max(Nmax_slice, surface["N_slice"])

    # =========================================================================
    # Other parameters needed to set up MC/DC object types
    # =========================================================================

    # Flags
    iQMC = input_card.technique["iQMC"]

    # Numbers
    N_tally_scores = input_card.setting["N_sensitivity"] + 1

    # =========================================================================
    # Default cards, if not given
    # =========================================================================

    # Default root universe
    if N_universe == 1:
        Nmax_cell = N_cell
        card = input_card.universes[0]
        card["N_cell"] = N_cell
        card["cell_IDs"] = np.arange(N_cell)

    # =========================================================================
    # Make types
    # =========================================================================

    type_.make_type_particle(iQMC, G)
    type_.make_type_particle_record(iQMC, G)
    type_.make_type_nuclide(G, J)
    type_.make_type_material(G, J, Nmax_nuclide)
    type_.make_type_surface(Nmax_slice)
    type_.make_type_cell(Nmax_surface)
    type_.make_type_universe(Nmax_cell)
    type_.make_type_lattice(input_card.lattices)
    type_.make_type_source(G)
    type_.make_type_tally(N_tally_scores, input_card.tally)
    type_.make_type_technique(N_particle, G, input_card.technique)
    type_.make_type_global(input_card)

    # =========================================================================
    # Make the global variable container
    #   TODO: Better alternative?
    # =========================================================================

    mcdc = np.zeros(1, dtype=type_.global_)[0]

    # =========================================================================
    # Nuclides
    # =========================================================================

    for i in range(N_nuclide):
        for name in type_.nuclide.names:
            mcdc["nuclides"][i][name] = input_card.nuclides[i][name]

    # =========================================================================
    # Materials
    # =========================================================================

    for i in range(N_material):
        for name in type_.material.names:
            mcdc["materials"][i][name] = input_card.materials[i][name]

    # =========================================================================
    # Surfaces
    # =========================================================================

    for i in range(N_surface):
        for name in type_.surface.names:
            if name not in ["J", "t"]:
                mcdc["surfaces"][i][name] = input_card.surfaces[i][name]

        # Variables with possible different sizes
        for name in ["J", "t"]:
            N = len(input_card.surfaces[i][name])
            mcdc["surfaces"][i][name][:N] = input_card.surfaces[i][name]

    # =========================================================================
    # Cells
    # =========================================================================

    for i in range(N_cell):
        for name in type_.cell.names:
            if name not in ["surface_IDs", "positive_flags"]:
                mcdc["cells"][i][name] = input_card.cells[i][name]

        # Variables with possible different sizes
        for name in ["surface_IDs", "positive_flags"]:
            N = len(input_card.cells[i][name])
            mcdc["cells"][i][name][:N] = input_card.cells[i][name]

    # =========================================================================
    # Universes
    # =========================================================================

    for i in range(N_universe):
        for name in type_.universe.names:
            if name not in ["cell_IDs"]:
                mcdc["universes"][i][name] = input_card.universes[i][name]

        # Variables with possible different sizes
        for name in ["cell_IDs"]:
            N = mcdc["universes"][i]["N_cell"]
            mcdc["universes"][i][name][:N] = input_card.universes[i][name]

    # =========================================================================
    # Lattices
    # =========================================================================

    for i in range(N_lattice):
        # Mesh
        for name in type_.mesh_uniform.names:
            mcdc["lattices"][i]["mesh"][name] = input_card.lattices[i]["mesh"][name]

        # Universe IDs
        Nx = mcdc["lattices"][i]["mesh"]["Nx"]
        Ny = mcdc["lattices"][i]["mesh"]["Ny"]
        Nz = mcdc["lattices"][i]["mesh"]["Nz"]
        mcdc["lattices"][i]["universe_IDs"][:Nx, :Ny, :Nz] = input_card.lattices[i][
            "universe_IDs"
        ]

    # =========================================================================
    # Source
    # =========================================================================

    for i in range(N_source):
        for name in type_.source.names:
            mcdc["sources"][i][name] = input_card.sources[i][name]

    # Normalize source probabilities
    tot = 0.0
    for S in mcdc["sources"]:
        tot += S["prob"]
    for S in mcdc["sources"]:
        S["prob"] /= tot

    # =========================================================================
    # Tally
    # =========================================================================

    for name in type_.tally.names:
        if name not in ["score", "mesh"]:
            mcdc["tally"][name] = input_card.tally[name]
    # Set mesh
    for name in type_.mesh_names:
        mcdc["tally"]["mesh"][name] = input_card.tally["mesh"][name]

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        mcdc["setting"][name] = input_card.setting[name]

    # Check if time boundary is above the final tally mesh time grid
    if mcdc["setting"]["time_boundary"] > mcdc["tally"]["mesh"]["t"][-1]:
        mcdc["setting"]["time_boundary"] = mcdc["tally"]["mesh"]["t"][-1]

    # =========================================================================
    # Technique
    # =========================================================================

    # Flags
    for name in [
        "weighted_emission",
        "implicit_capture",
        "population_control",
        "weight_window",
        "weight_roulette",
        "iQMC",
        "IC_generator",
        "time_census",
        "branchless_collision",
    ]:
        mcdc["technique"][name] = input_card.technique[name]

    # =========================================================================
    # Population control
    # =========================================================================

    # Population control technique (PCT)
    mcdc["technique"]["pct"] = input_card.technique["pct"]
    mcdc["technique"]["pc_factor"] = input_card.technique["pc_factor"]

    # =========================================================================
    # Weight window (WW)
    # =========================================================================

    # WW mesh
    for name in type_.mesh_names[:-1]:
        mcdc["technique"]["ww_mesh"][name] = input_card.technique["ww_mesh"][name]

    # WW windows
    mcdc["technique"]["ww"] = input_card.technique["ww"]
    mcdc["technique"]["ww_width"] = input_card.technique["ww_width"]

    # =========================================================================
    # Weight roulette
    # =========================================================================

    # Threshold
    mcdc["technique"]["wr_threshold"] = input_card.technique["wr_threshold"]

    # Survival probability
    mcdc["technique"]["wr_chance"] = input_card.technique["wr_chance"]

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

    for name in type_.technique.names:
        if name[:4] == "iqmc":
            if name not in [
                "iqmc_flux_old",
                "iqmc_flux_outter",
                "iqmc_mesh",
                "iqmc_source",
                "iqmc_res",
                "iqmc_lds",
                "iqmc_generator",
                "iqmc_sweep_counter",
            ]:
                mcdc["technique"][name] = input_card.technique[name]

    if input_card.technique["iQMC"]:
        mcdc["technique"]["iqmc_mesh"]["x"] = input_card.technique["iqmc_mesh"]["x"]
        mcdc["technique"]["iqmc_mesh"]["y"] = input_card.technique["iqmc_mesh"]["y"]
        mcdc["technique"]["iqmc_mesh"]["z"] = input_card.technique["iqmc_mesh"]["z"]
        mcdc["technique"]["iqmc_mesh"]["t"] = input_card.technique["iqmc_mesh"]["t"]
        mcdc["technique"]["iqmc_generator"] = input_card.technique["iqmc_generator"]
        # generate lds
        if input_card.technique["iqmc_generator"] == "sobol":
            scramble = mcdc["technique"]["iqmc_scramble"]
            N_dim = mcdc["technique"]["iqmc_N_dim"]
            N = mcdc["setting"]["N_particle"]
            sampler = qmc.Sobol(d=N_dim, scramble=scramble)
            m = math.ceil(math.log(N, 2))
            mcdc["setting"]["N_particle"] = 2**m
            mcdc["technique"]["iqmc_lds"] = sampler.random_base2(m=m)
            # first row of an unscrambled Sobol sequence is all zeros
            # and throws off some of the algorithms. So we add small amount
            # to avoid this issue
            # mcdc["technique"]["lds"][0,:] += 1e-6
            mcdc["technique"]["iqmc_lds"][mcdc["technique"]["iqmc_lds"] == 0.0] += 1e-6
            # lds is shape (2**m, d)
        if input_card.technique["iqmc_generator"] == "halton":
            scramble = mcdc["technique"]["iqmc_scramble"]
            N_dim = mcdc["technique"]["iqmc_N_dim"]
            seed = mcdc["technique"]["iqmc_seed"]
            N = mcdc["setting"]["N_particle"]
            sampler = qmc.Halton(d=N_dim, scramble=scramble, seed=seed)
            sampler.fast_forward(1)
            mcdc["technique"]["iqmc_lds"] = sampler.random(N)
        if input_card.technique["iqmc_generator"] == "random":
            seed = mcdc["technique"]["iqmc_seed"]
            N_dim = mcdc["technique"]["iqmc_N_dim"]
            N = mcdc["setting"]["N_particle"]
            np.random.seed(seed)
            mcdc["technique"]["iqmc_lds"] = np.random.random((N, N_dim))

    # =========================================================================
    # Time census
    # =========================================================================

    # Census time
    mcdc["technique"]["census_time"] = input_card.technique["census_time"]

    # Census index
    mcdc["technique"]["census_idx"] = 0

    # =========================================================================
    # RNG
    # =========================================================================

    # RNG seed and stride
    mcdc["rng_seed_base"] = mcdc["setting"]["rng_seed"]
    mcdc["rng_seed"] = mcdc["setting"]["rng_seed"]
    mcdc["rng_stride"] = mcdc["setting"]["rng_stride"]

    # =========================================================================
    # MPI
    # =========================================================================

    # MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Distribute work to MPI ranks
    kernel.distribute_work(mcdc["setting"]["N_particle"], mcdc)

    # =========================================================================
    # Particle banks
    # =========================================================================

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"

    # IC generator banks
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"

    # =========================================================================
    # Eigenvalue (or fixed-source)
    # =========================================================================

    # Initial guess
    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # Activate tally scoring for fixed-source
    if not mcdc["setting"]["mode_eigenvalue"]:
        mcdc["cycle_active"] = True

    # All active eigenvalue cycle?
    elif mcdc["setting"]["N_inactive"] == 0:
        mcdc["cycle_active"] = True

    # =========================================================================
    # Source file
    # =========================================================================

    if mcdc["setting"]["source_file"]:
        with h5py.File(mcdc["setting"]["source_file_name"], "r") as f:
            # Get source particle size
            N_particle = f["particles_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["particles"][start:end]
            mcdc["bank_source"]["size"] = N_local

    # =========================================================================
    # IC file
    # =========================================================================

    if mcdc["setting"]["IC_file"]:
        with h5py.File(mcdc["setting"]["IC_file_name"], "r") as f:
            # =================================================================
            # Set neutron source
            # =================================================================

            # Get source particle size
            N_particle = f["IC/neutrons_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["IC/neutrons"][start:end]
            mcdc["bank_source"]["size"] = N_local

            # =================================================================
            # Set precursor source
            # =================================================================

            # Get source particle size
            N_precursor = f["IC/precursors_size"][()]

            # Redistribute work
            kernel.distribute_work(N_precursor, mcdc, True)  # precursor = True
            N_local = mcdc["mpi_work_size_precursor"]
            start = mcdc["mpi_work_start_precursor"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_precursor"]["precursors"][:N_local] = f["IC/precursors"][
                start:end
            ]
            mcdc["bank_precursor"]["size"] = N_local

            # Set precursor strength
            if N_precursor > 0 and N_particle > 0:
                mcdc["precursor_strength"] = mcdc["bank_precursor"]["precursors"][0][
                    "w"
                ]


def dictlist_to_h5group(dictlist, input_group, name):
    main_group = input_group.create_group(name + "s")
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        else:
            group[k] = v


def generate_hdf5():
    if mcdc["mpi_master"]:
        if mcdc["setting"]["progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["setting"]["output"] + ".h5", "w") as f:
            # Input card
            if mcdc["setting"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                dictlist_to_h5group(input_card.nuclides, input_group, "nuclide")
                dictlist_to_h5group(input_card.materials, input_group, "material")
                dictlist_to_h5group(input_card.surfaces, input_group, "surface")
                dictlist_to_h5group(input_card.cells, input_group, "cell")
                dictlist_to_h5group(input_card.universes, input_group, "universe")
                dictlist_to_h5group(input_card.lattices, input_group, "lattice")
                dictlist_to_h5group(input_card.sources, input_group, "source")
                dict_to_h5group(input_card.tally, input_group.create_group("tally"))
                dict_to_h5group(input_card.setting, input_group.create_group("setting"))
                dict_to_h5group(
                    input_card.technique, input_group.create_group("technique")
                )

            # Tally
            T = mcdc["tally"]
            f.create_dataset("tally/grid/t", data=T["mesh"]["t"])
            f.create_dataset("tally/grid/x", data=T["mesh"]["x"])
            f.create_dataset("tally/grid/y", data=T["mesh"]["y"])
            f.create_dataset("tally/grid/z", data=T["mesh"]["z"])
            f.create_dataset("tally/grid/mu", data=T["mesh"]["mu"])
            f.create_dataset("tally/grid/azi", data=T["mesh"]["azi"])
            f.create_dataset("tally/grid/g", data=T["mesh"]["g"])

            # Scores
            for name in T["score"].dtype.names:
                if mcdc["tally"][name]:
                    name_h5 = name.replace("_", "-")
                    f.create_dataset(
                        "tally/" + name_h5 + "/mean",
                        data=np.squeeze(T["score"][name]["mean"]),
                    )
                    f.create_dataset(
                        "tally/" + name_h5 + "/sdev",
                        data=np.squeeze(T["score"][name]["sdev"]),
                    )

            # Eigenvalues
            if mcdc["setting"]["mode_eigenvalue"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                else:
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["setting"]["gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # dump iQMC mesh
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc_mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc_mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc_mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc_mesh"]["z"])
                f.create_dataset("iqmc/material_idx", data=T["iqmc_material_idx"])
                # dump x,y,z scalar flux across all groups
                f.create_dataset("iqmc/flux", data=np.squeeze(T["iqmc_flux"]))
                # iteration data
                f.create_dataset("iqmc/itteration_count", data=T["iqmc_itt"])
                f.create_dataset("iqmc/final_residual", data=T["iqmc_res"])
                f.create_dataset("iqmc/sweep_count", data=T["iqmc_sweep_counter"])
                if mcdc["setting"]["mode_eigenvalue"]:
                    f.create_dataset(
                        "iqmc/outter_itteration_count", data=T["iqmc_itt_outter"]
                    )
                    f.create_dataset(
                        "iqmc/outter_final_residual", data=T["iqmc_res_outter"]
                    )

            # Particle tracker
            if mcdc["setting"]["track_particle"]:
                with h5py.File(mcdc["setting"]["output"] + "_ptrack.h5", "w") as f:
                    N_track = mcdc["particle_track_N"]
                    f.create_dataset("tracks", data=mcdc["particle_track"][:N_track])

            # IC generator
            if mcdc["technique"]["IC_generator"]:
                Nn = mcdc["technique"]["IC_bank_neutron"]["size"]
                Np = mcdc["technique"]["IC_bank_precursor"]["size"]
                f.create_dataset(
                    "IC/neutrons",
                    data=mcdc["technique"]["IC_bank_neutron"]["particles"][:Nn],
                )
                f.create_dataset(
                    "IC/precursors",
                    data=mcdc["technique"]["IC_bank_precursor"]["precursors"][:Np],
                )
                f.create_dataset("IC/neutrons_size", data=Nn)
                f.create_dataset("IC/precursors_size", data=Np)
                f.create_dataset(
                    "IC/fission", data=mcdc["technique"]["IC_fission"] / Nn
                )

    # Save particle?
    if mcdc["setting"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def closeout():
    # Runtime
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["setting"]["output"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

    print_runtime(mcdc)
    input_card.reset()
