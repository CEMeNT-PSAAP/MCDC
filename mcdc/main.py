import h5py
import numpy as np

from mpi4py import MPI

import mcdc.kernel as kernel
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.loop     import loop_main
from mcdc.print_   import print_banner, print_msg, print_runtime,\
                          print_header_eigenvalue
from mcdc.util     import profile

# Get input_card and set global variables as "mcdc"
import mcdc.global_ as mcdc_
input_card = mcdc_.input_card
mcdc       = mcdc_.global_

#@profile
def run():
    # Print banner and hardware configuration
    print_banner()

    # Preparation:
    #   process input cards, make types, and allocate global variables
    prepare()
    input_card.reset()
    
    # Run
    print_msg(" Now running TNT...")
    if mcdc['setting']['mode_eigenvalue']:
        print_header_eigenvalue(mcdc)
    MPI.COMM_WORLD.Barrier()
    mcdc['runtime_total'] = MPI.Wtime()
    loop_main(mcdc)
    MPI.COMM_WORLD.Barrier()
    mcdc['runtime_total'] = MPI.Wtime() - mcdc['runtime_total']
    
    # Output: generate hdf5 output files
    generate_hdf5()

    # Closout
    print_runtime(mcdc)

def prepare():
    global mcdc
    
    print_msg("\n Preparing...")

    # =========================================================================
    # Some numbers
    # =========================================================================

    G          = input_card.materials[0]['G']
    J          = input_card.materials[0]['J']
    N_material = len(input_card.materials)
    N_surface  = len(input_card.surfaces)
    N_cell     = len(input_card.cells)
    N_universe = len(input_card.universes)
    N_lattice  = len(input_card.lattices)
    N_source   = len(input_card.sources)
    N_particle = input_card.setting['N_particle']
    N_cycle    = input_card.setting['N_cycle']
    # Derived numbers
    Nmax_surface = 0
    Nmax_slice   = 0 # Surface time-dependence slice
    Nmax_cell    = 0
    for surface in input_card.surfaces:
        Nmax_slice = max(Nmax_slice, surface['N_slice'])
    for cell in input_card.cells:
        Nmax_surface = max(Nmax_surface, cell['N_surface'])
    for universe in input_card.universes:
        Nmax_cell = max(Nmax_cell, universe['N_cell'])

    # =========================================================================
    # Default cards, if not given
    # =========================================================================

    # Default root universe
    if N_universe == 1:
        Nmax_cell        = N_cell
        card             = input_card.universes[0]
        card['N_cell']   = N_cell
        card['cell_IDs'] = np.arange(N_cell)

    # =========================================================================
    # Make types
    # =========================================================================

    type_.make_type_material(G,J)
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
    # Materials
    # =========================================================================

    for i in range(N_material):
        for name in type_.material.names:
            mcdc['materials'][i][name] = input_card.materials[i][name]
    
    # =========================================================================
    # Surfaces
    # =========================================================================

    for i in range(N_surface):
        for name in type_.surface.names:
            if name in ['J', 't']:
                shape = mcdc['surfaces'][i][name].shape
                input_card.surfaces[i][name].resize(shape)
            mcdc['surfaces'][i][name] = input_card.surfaces[i][name]
    
    # =========================================================================
    # Cells
    # =========================================================================

    for i in range(N_cell):
        for name in type_.cell.names:
            if name in ['surface_IDs', 'positive_flags']:
                N = mcdc['cells'][i]['N_surface']
                mcdc['cells'][i][name][:N] = input_card.cells[i][name]
            else:
                mcdc['cells'][i][name] = input_card.cells[i][name]

    # =========================================================================
    # Universes
    # =========================================================================

    for i in range(N_universe):
        for name in type_.universe.names:
            if name in ['cell_IDs']:
                N = mcdc['universes'][i]['N_cell']
                mcdc['universes'][i][name][:N] = input_card.universes[i][name]
            else:
                mcdc['universes'][i][name] = input_card.universes[i][name]

    # =========================================================================
    # Lattices
    # =========================================================================

    for i in range(N_lattice):
        # Mesh
        mcdc['lattices'][i]['mesh']['x0'] = input_card.lattices[i]['mesh']['x0']
        mcdc['lattices'][i]['mesh']['dx'] = input_card.lattices[i]['mesh']['dx']
        mcdc['lattices'][i]['mesh']['Nx'] = input_card.lattices[i]['mesh']['Nx']
        mcdc['lattices'][i]['mesh']['y0'] = input_card.lattices[i]['mesh']['y0']
        mcdc['lattices'][i]['mesh']['dy'] = input_card.lattices[i]['mesh']['dy']
        mcdc['lattices'][i]['mesh']['Ny'] = input_card.lattices[i]['mesh']['Ny']
        mcdc['lattices'][i]['mesh']['z0'] = input_card.lattices[i]['mesh']['z0']
        mcdc['lattices'][i]['mesh']['dz'] = input_card.lattices[i]['mesh']['dz']
        mcdc['lattices'][i]['mesh']['Nz'] = input_card.lattices[i]['mesh']['Nz']

        # Universe IDs
        Nx = input_card.lattices[i]['mesh']['Nx']
        Ny = input_card.lattices[i]['mesh']['Ny']
        Nz = input_card.lattices[i]['mesh']['Nz']
        mcdc['lattices'][i]['universe_IDs'][:Nx,:Ny,:Nz] =\
            input_card.lattices[i]['universe_IDs']

    # =========================================================================
    # Source
    # =========================================================================

    for i in range(N_source):
        for name in type_.source.names:
            mcdc['sources'][i][name] = input_card.sources[i][name]
    
    # Normalize source probabilities
    tot = 0.0
    for S in mcdc['sources']:
        tot += S['prob']
    for S in mcdc['sources']:
        S['prob'] /= tot

    # =========================================================================
    # Tally
    # =========================================================================

    for name in type_.tally.names:
        if name not in ['score', 'mesh']:
            mcdc['tally'][name] = input_card.tally[name]
    
    # Set mesh
    mcdc['tally']['mesh']['x']   = input_card.tally['mesh']['x']
    mcdc['tally']['mesh']['y']   = input_card.tally['mesh']['y']
    mcdc['tally']['mesh']['z']   = input_card.tally['mesh']['z']
    mcdc['tally']['mesh']['t']   = input_card.tally['mesh']['t']
    mcdc['tally']['mesh']['mu']  = input_card.tally['mesh']['mu']
    mcdc['tally']['mesh']['azi'] = input_card.tally['mesh']['azi']

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        mcdc['setting'][name] = input_card.setting[name]

    # Check if time boundary matches the final tally mesh time grid
    if mcdc['setting']['time_boundary'] > mcdc['tally']['mesh']['t'][-1]:
        mcdc['setting']['time_boundary'] = mcdc['tally']['mesh']['t'][-1]
    
    # =========================================================================
    # Technique
    # =========================================================================

    for name in type_.technique.names:
        if name not in ['ww_mesh',
                        'census_idx',
                        'IC_bank_neutron', 'IC_bank_precursor',
                        'IC_bank_neutron_local', 'IC_bank_precursor_local',
                        'IC_tally_n', 'IC_tally_C', 'IC_n_eff', 'IC_C_eff',
                        'IC_Pmax_n', 'IC_Pmax_C', 'IC_resample']:
            mcdc['technique'][name] = input_card.technique[name]

    # Set time census parameter
    if mcdc['technique']['time_census']:
        mcdc['technique']['census_idx'] = 0

    # Set weight window mesh
    if input_card.technique['weight_window']:
        name = 'ww_mesh'
        mcdc['technique'][name]['x']   = input_card.technique[name]['x']
        mcdc['technique'][name]['y']   = input_card.technique[name]['y']
        mcdc['technique'][name]['z']   = input_card.technique[name]['z']
        mcdc['technique'][name]['t']   = input_card.technique[name]['t']
        mcdc['technique'][name]['mu']  = input_card.technique[name]['mu']
        mcdc['technique'][name]['azi'] = input_card.technique[name]['azi']

    # =========================================================================
    # Global tally
    # =========================================================================

    mcdc['k_eff'] = mcdc['setting']['k_init']

    # =========================================================================
    # Misc.
    # =========================================================================

    # RNG seed and stride
    mcdc['rng_seed_base'] = mcdc['setting']['rng_seed']
    mcdc['rng_seed']      = mcdc['setting']['rng_seed']
    mcdc['rng_stride']    = mcdc['setting']['rng_stride']

    # Set MPI parameters
    mcdc['mpi_size']   = MPI.COMM_WORLD.Get_size()
    mcdc['mpi_rank']   = MPI.COMM_WORLD.Get_rank()
    mcdc['mpi_master'] = mcdc['mpi_rank'] == 0

    # Particle bank tags
    mcdc['bank_active']['tag'] = 'active'
    mcdc['bank_census']['tag'] = 'census'
    mcdc['bank_source']['tag'] = 'source'

    # Distribute work to processors
    kernel.distribute_work(mcdc['setting']['N_particle'], mcdc)

    # TODO: Set source bank if using filed source
    '''
    if mcdc['setting']['filed_source']:
        start = mcdc['mpi_work_start']
        end   = start + mcdc['mpi_work_size']
        # Load particles from file
        with h5py.File(mcdc['setting']['source_file'], 'r') as f:
            particles = f['IC/particles'][start:end]
        for P in particles:
            kernel.add_particle(P, mcdc['bank_source'])
    '''

    # Activate tally scoring for fixed-source
    if not mcdc['setting']['mode_eigenvalue']:
        mcdc['cycle_active'] = True

def generate_hdf5():
    if mcdc['mpi_master']:
        if mcdc['setting']['progress_bar']: print_msg('')
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc['setting']['output']+'.h5', 'w') as f:
            # Runtime
            for name in ['total', 'bank_management']:
                f.create_dataset("runtime_"+name,
                                 data=np.array([mcdc['runtime_'+name]]))

            # Tally
            T = mcdc['tally']
            f.create_dataset("tally/grid/t", data=T['mesh']['t'])
            f.create_dataset("tally/grid/x", data=T['mesh']['x'])
            f.create_dataset("tally/grid/y", data=T['mesh']['y'])
            f.create_dataset("tally/grid/z", data=T['mesh']['z'])
            f.create_dataset("tally/grid/mu", data=T['mesh']['mu'])
            f.create_dataset("tally/grid/azi", data=T['mesh']['azi'])
            
            # Scores
            for name in T['score'].dtype.names:
                if mcdc['tally'][name]:
                    name_h5 = name.replace('_','-')
                    f.create_dataset("tally/"+name_h5+"/mean",
                                     data=np.squeeze(T['score'][name]['mean']))
                    f.create_dataset("tally/"+name_h5+"/sdev",
                                     data=np.squeeze(T['score'][name]['sdev']))
                
            # Eigenvalues
            if mcdc['setting']['mode_eigenvalue']:
                N_cycle = mcdc['setting']['N_cycle']
                f.create_dataset("k_cycle",data=mcdc['k_cycle'][:N_cycle])
                f.create_dataset("k_mean",data=mcdc['k_avg_running'])
                f.create_dataset("k_sdev",data=mcdc['k_sdv_running'])
                if mcdc['setting']['gyration_radius']:
                    f.create_dataset("gyration_radius",data=mcdc['gyration_radius'][:N_cycle])

            # IC generator
            if mcdc['technique']['IC_generator']:
                Nn = mcdc['technique']['IC_bank_neutron']['size']
                Np = mcdc['technique']['IC_bank_precursor']['size']
                f.create_dataset("IC/neutron",data=mcdc['technique']['IC_bank_neutron']['content'][:Nn])
                f.create_dataset("IC/precursor",data=mcdc['technique']['IC_bank_precursor']['content'][:Np])
