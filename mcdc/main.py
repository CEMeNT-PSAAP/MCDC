import h5py
import numpy as np

from mpi4py import MPI

import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.looper   import loop_simulation
from mcdc.print_   import print_banner, print_msg, print_runtime
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
        print_msg(" #     k-eig    GyRad.")
        print_msg(" ====  ======   ======")
    mcdc['runtime_total'] = MPI.Wtime()
    loop_simulation(mcdc)
    mcdc['runtime_total'] = MPI.Wtime() - mcdc['runtime_total']
    
    # Output: generate hdf5 output file
    if mcdc['mpi_master']: generate_hdf5()

    # Closout
    print_runtime(mcdc)

def prepare():
    global mcdc
    
    print_msg("\n Preparing...")

    # Some numbers
    G           = input_card.materials[0]['G']
    J           = input_card.materials[0]['J']
    N_materials = len(input_card.materials)
    N_surfaces  = len(input_card.surfaces)
    N_cells     = len(input_card.cells)
    N_universes = len(input_card.universes)
    N_sources   = len(input_card.sources)
    N_iter      = input_card.setting['N_iter']
    N_hist      = input_card.setting['N_hist']
    # Derived numbers
    Nmax_surfaces = 0
    for cell in input_card.cells:
        Nmax_surfaces = max(Nmax_surfaces, cell['N_surfaces'])
    Nmax_cells    = 0
    for universe in input_card.universes:
        Nmax_cells = max(Nmax_cells, universe['N_cells'])

    # Default universe card if not given
    if N_universes == 0:
        Nmax_cells  = N_cells
        N_universes = 1
        card = {'ID' : 0, 'N_cells' : N_cells, 'cell_IDs' : np.arange(N_cells)}
        input_card.universes.append(card)

    # Make types
    type_.make_type_material(G,J)
    type_.make_type_cell(Nmax_surfaces)
    type_.make_type_universe(Nmax_cells)
    type_.make_type_lattice(input_card.lattice)
    type_.make_type_source(G)
    type_.make_type_tally(input_card.tally, G, N_iter)
    type_.make_type_technique(input_card.technique)
    type_.make_type_global(input_card)

    # The global variable container
    mcdc = np.zeros(1, dtype=type_.global_)[0]
    
    # Material
    for i in range(N_materials):
        for name in type_.material.names:
            mcdc['materials'][i][name] = input_card.materials[i][name]
    
    # Surface
    for i in range(N_surfaces):
        for name in type_.surface.names:
            mcdc['surfaces'][i][name] = input_card.surfaces[i][name]
    
    # Cell
    for i in range(N_cells):
        for name in type_.cell.names:
            mcdc['cells'][i][name] = input_card.cells[i][name]

    # Universe
    for i in range(N_universes):
        for name in type_.universe.names:
            mcdc['universes'][i][name] = input_card.universes[i][name]

    # Lattice
    for name in type_.lattice.names:
        if name not in ['mesh']:
            mcdc['lattice'][name] = input_card.lattice[name]
    # Set mesh
    mcdc['lattice']['mesh']['x'] = input_card.lattice['mesh']['x']
    mcdc['lattice']['mesh']['y'] = input_card.lattice['mesh']['y']
    mcdc['lattice']['mesh']['z'] = input_card.lattice['mesh']['z']
    mcdc['lattice']['mesh']['t'] = input_card.lattice['mesh']['t']

    # Source
    for i in range(N_sources):
        for name in type_.source.names:
            mcdc['sources'][i][name] = input_card.sources[i][name]
    # Normalize source probabilities
    tot = 0.0
    for S in mcdc['sources']:
        tot += S['prob']
    for S in mcdc['sources']:
        S['prob'] /= tot

    # Tally
    for name in type_.tally.names:
        if name not in ['score', 'mesh']:
            mcdc['tally'][name] = input_card.tally[name]
    # Set mesh
    mcdc['tally']['mesh']['x'] = input_card.tally['mesh']['x']
    mcdc['tally']['mesh']['y'] = input_card.tally['mesh']['y']
    mcdc['tally']['mesh']['z'] = input_card.tally['mesh']['z']
    mcdc['tally']['mesh']['t'] = input_card.tally['mesh']['t']

    # Setting
    for name in type_.setting.names:
        mcdc['setting'][name] = input_card.setting[name]
    # Check if time boundary matches the final tally mesh time grid
    if mcdc['setting']['time_boundary'] > mcdc['tally']['mesh']['t'][-1]:
        mcdc['setting']['time_boundary'] = mcdc['tally']['mesh']['t'][-1]
    
    # Technique
    for name in type_.technique.names:
        if name not in ['ww_mesh']:
            mcdc['technique'][name] = input_card.technique[name]
    # Set weight window mesh
    if input_card.technique['weight_window']:
        mcdc['technique']['ww_mesh']['x'] = input_card.technique['ww_mesh']['x']
        mcdc['technique']['ww_mesh']['y'] = input_card.technique['ww_mesh']['y']
        mcdc['technique']['ww_mesh']['z'] = input_card.technique['ww_mesh']['z']
        mcdc['technique']['ww_mesh']['t'] = input_card.technique['ww_mesh']['t']

    # Global tally
    mcdc['k_eff']     = 1.0
    mcdc['alpha_eff'] = 0.0
    if mcdc['setting']['mode_eigenvalue']:
        mcdc['k_eff']     = mcdc['setting']['k_init']
        mcdc['k_iterate'] = np.zeros(N_iter, dtype=np.float64)
        mcdc['gyration_radius'] = np.zeros(N_iter, dtype=np.float64)
        if mcdc['setting']['mode_alpha']:
            mcdc['alpha_eff']     = mcdc['setting']['alpha_init']
            mcdc['alpha_iterate'] = np.zeros(N_iter, dtype=np.float64)
        # Gyration radius type
        if input_card.setting['gyration_all']:
            mcdc['gyration_all'] = True
        elif input_card.setting['gyration_infinite_z']:
            mcdc['gyration_infinite_z'] = True
        elif input_card.setting['gyration_only_x']:
            mcdc['gyration_only_x'] = True

    # RNG seed and stride
    mcdc['rng_seed_base'] = mcdc['setting']['rng_seed']
    mcdc['rng_seed']      = mcdc['setting']['rng_seed']
    mcdc['rng_g']         = 2806196910506780709
    mcdc['rng_c']         = 1
    mcdc['rng_mod']       = 2**63
    mcdc['rng_stride']    = mcdc['setting']['rng_stride']

    # Set MPI parameters
    mcdc['mpi_size']   = MPI.COMM_WORLD.Get_size()
    mcdc['mpi_rank']   = MPI.COMM_WORLD.Get_rank()
    mcdc['mpi_master'] = mcdc['mpi_rank'] == 0
    
def generate_hdf5():
    if mcdc['setting']['progress_bar']: print_msg('')
    print_msg(" Generating tally HDF5 files...")

    with h5py.File(mcdc['setting']['output']+'.h5', 'w') as f:
        # Runtime
        f.create_dataset("runtime",data=np.array([mcdc['runtime_total']]))

        # Tally
        T = mcdc['tally']
        f.create_dataset("tally/grid/t", data=T['mesh']['t'])
        f.create_dataset("tally/grid/x", data=T['mesh']['x'])
        f.create_dataset("tally/grid/y", data=T['mesh']['y'])
        f.create_dataset("tally/grid/z", data=T['mesh']['z'])
        
        # Scores
        for name in T['score'].dtype.names:
            name_h5 = name.replace('_','-')
            f.create_dataset("tally/"+name_h5+"/mean",
                             data=np.squeeze(T['score'][name]['mean']))
            f.create_dataset("tally/"+name_h5+"/sdev",
                             data=np.squeeze(T['score'][name]['sdev']))
            T['score'][name]['mean'].fill(0.0)
            T['score'][name]['sdev'].fill(0.0)
            
        # Eigenvalues
        if mcdc['setting']['mode_eigenvalue']:
            f.create_dataset("keff",data=mcdc['k_iterate'])
            f.create_dataset("gyration_radius",data=mcdc['gyration_radius'])
            mcdc['k_iterate'].fill(0.0)
            mcdc['gyration_radius'].fill(0.0)
            if mcdc['setting']['mode_alpha']:
                f.create_dataset("alpha_eff",data=mcdc['alpha_iterate'])
                mcdc['alpha_iterate'].fill(0.0)
