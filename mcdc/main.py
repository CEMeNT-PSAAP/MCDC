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
    # TODO: add if iQMC execute iqmc_loop()
    simulation_start = MPI.Wtime()

    # iQMC loop
    if mcdc["technique"]["iQMC"]:
        loop_iqmc(mcdc)
    else:
        loop_main(mcdc)
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

    # Maximum object reference
    Nmax_nuclide = 0
    Nmax_surface = 0
    Nmax_cell = 0
    for material in input_card.materials:
        Nmax_nuclide = max(Nmax_nuclide, material["N_nuclide"])
    for cell in input_card.cells:
        Nmax_surface = max(Nmax_surface, cell["N_surface"])
    for universe in input_card.universes:
        Nmax_cell = max(Nmax_cell, universe["N_cell"])

    # Maximum time-dependent surface slices
    Nmax_slice = 0
    for surface in input_card.surfaces:
        Nmax_slice = max(Nmax_slice, surface["N_slice"])

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

    type_.make_type_particle(input_card)
    type_.make_type_particle_record(input_card)
    type_.make_type_nuclide(G, J)
    type_.make_type_material(G, J, Nmax_nuclide)
    type_.make_type_surface(Nmax_slice)
    type_.make_type_cell(Nmax_surface)
    type_.make_type_universe(Nmax_cell)
    type_.make_type_lattice(input_card.lattices)
    type_.make_type_source(G)
    type_.make_type_tally(input_card)
    type_.make_type_technique(input_card)
    type_.make_type_global(input_card)

    # The global variable container
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
            if name in ["J", "t"]:
                shape = mcdc["surfaces"][i][name].shape
                input_card.surfaces[i][name].resize(shape)
            mcdc["surfaces"][i][name] = input_card.surfaces[i][name]

    # =========================================================================
    # Cells
    # =========================================================================

    for i in range(N_cell):
        for name in type_.cell.names:
            if name in ["surface_IDs", "positive_flags"]:
                N = mcdc["cells"][i]["N_surface"]
                mcdc["cells"][i][name][:N] = input_card.cells[i][name]
            else:
                mcdc["cells"][i][name] = input_card.cells[i][name]

    # =========================================================================
    # Universes
    # =========================================================================

    for i in range(N_universe):
        for name in type_.universe.names:
            if name in ["cell_IDs"]:
                N = mcdc["universes"][i]["N_cell"]
                mcdc["universes"][i][name][:N] = input_card.universes[i][name]
            else:
                mcdc["universes"][i][name] = input_card.universes[i][name]

    # =========================================================================
    # Lattices
    # =========================================================================

    for i in range(N_lattice):
        # Mesh
        mcdc["lattices"][i]["mesh"]["x0"] = input_card.lattices[i]["mesh"]["x0"]
        mcdc["lattices"][i]["mesh"]["dx"] = input_card.lattices[i]["mesh"]["dx"]
        mcdc["lattices"][i]["mesh"]["Nx"] = input_card.lattices[i]["mesh"]["Nx"]
        mcdc["lattices"][i]["mesh"]["y0"] = input_card.lattices[i]["mesh"]["y0"]
        mcdc["lattices"][i]["mesh"]["dy"] = input_card.lattices[i]["mesh"]["dy"]
        mcdc["lattices"][i]["mesh"]["Ny"] = input_card.lattices[i]["mesh"]["Ny"]
        mcdc["lattices"][i]["mesh"]["z0"] = input_card.lattices[i]["mesh"]["z0"]
        mcdc["lattices"][i]["mesh"]["dz"] = input_card.lattices[i]["mesh"]["dz"]
        mcdc["lattices"][i]["mesh"]["Nz"] = input_card.lattices[i]["mesh"]["Nz"]

        # Universe IDs
        Nx = input_card.lattices[i]["mesh"]["Nx"]
        Ny = input_card.lattices[i]["mesh"]["Ny"]
        Nz = input_card.lattices[i]["mesh"]["Nz"]
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
    mcdc["tally"]["mesh"]["x"] = input_card.tally["mesh"]["x"]
    mcdc["tally"]["mesh"]["y"] = input_card.tally["mesh"]["y"]
    mcdc["tally"]["mesh"]["z"] = input_card.tally["mesh"]["z"]
    mcdc["tally"]["mesh"]["t"] = input_card.tally["mesh"]["t"]
    mcdc["tally"]["mesh"]["mu"] = input_card.tally["mesh"]["mu"]
    mcdc["tally"]["mesh"]["azi"] = input_card.tally["mesh"]["azi"]

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        mcdc["setting"][name] = input_card.setting[name]

    # Check if time boundary matches the final tally mesh time grid
    if mcdc["setting"]["time_boundary"] > mcdc["tally"]["mesh"]["t"][-1]:
        mcdc["setting"]["time_boundary"] = mcdc["tally"]["mesh"]["t"][-1]

    # =========================================================================
    # Technique
    # =========================================================================

    for name in type_.technique.names:
        if name not in [
            "ww_mesh",
            "census_idx",
            "iqmc_flux_old",
            "iqmc_flux_outter",
            "iqmc_mesh",
            "iqmc_source",
            "lds",
            "iqmc_effective_scattering",
            "iqmc_effective_fission",
            "iqmc_res",
            "iqmc_generator",
            "wr_chance",
            "wr_threshold",
            "IC_bank_neutron",
            "IC_bank_precursor",
            "IC_bank_neutron_local",
            "IC_bank_precursor_local",
            "IC_fission_score",
            "IC_fission",
        ]:
            mcdc["technique"][name] = input_card.technique[name]

    # Set time census parameter
    if mcdc["technique"]["time_census"]:
        mcdc["technique"]["census_idx"] = 0

    # Set weight window mesh
    if input_card.technique["weight_window"]:
        name = "ww_mesh"
        mcdc["technique"][name]["x"] = input_card.technique[name]["x"]
        mcdc["technique"][name]["y"] = input_card.technique[name]["y"]
        mcdc["technique"][name]["z"] = input_card.technique[name]["z"]
        mcdc["technique"][name]["t"] = input_card.technique[name]["t"]
        mcdc["technique"][name]["mu"] = input_card.technique[name]["mu"]
        mcdc["technique"][name]["azi"] = input_card.technique[name]["azi"]

    # =========================================================================
    # Weight roulette
    # =========================================================================
    if input_card.technique["weight_roulette"]:
        mcdc["technique"]["wr_chance"] = input_card.technique["wr_chance"]
        mcdc["technique"]["wr_threshold"] = input_card.technique["wr_threshold"]

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

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
            mcdc["technique"]["lds"] = sampler.random_base2(m=m)
            # lds is shape (2**m, d)
        if input_card.technique["iqmc_generator"] == "halton":
            scramble = mcdc["technique"]["iqmc_scramble"]
            N_dim = mcdc["technique"]["iqmc_N_dim"]
            seed = mcdc["technique"]["iqmc_seed"]
            N = mcdc["setting"]["N_particle"]
            sampler = qmc.Halton(d=N_dim, scramble=scramble, seed=seed)
            sampler.fast_forward(1)
            mcdc["technique"]["lds"] = sampler.random(N)
        if input_card.technique["iqmc_generator"] == "random":
            seed = mcdc["technique"]["iqmc_seed"]
            N_dim = mcdc["technique"]["iqmc_N_dim"]
            N = mcdc["setting"]["N_particle"]
            np.random.seed(seed)
            mcdc["technique"]["lds"] = np.random.random((N, N_dim))

    # =========================================================================
    # Global tally
    # =========================================================================

    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # =========================================================================
    # Misc.
    # =========================================================================

    # RNG seed and stride
    mcdc["rng_seed_base"] = mcdc["setting"]["rng_seed"]
    mcdc["rng_seed"] = mcdc["setting"]["rng_seed"]
    mcdc["rng_stride"] = mcdc["setting"]["rng_stride"]

    # Set MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"

    # Distribute work to processors
    kernel.distribute_work(mcdc["setting"]["N_particle"], mcdc)

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
                # TODO: dump time dependent tallies
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc_mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc_mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc_mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc_mesh"]["z"])

                # dump x,y,z scalar flux across all groups
                f.create_dataset(
                    "tally/" + "iqmc_flux", data=np.squeeze(T["iqmc_flux"])
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
