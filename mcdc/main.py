import h5py

from   mcdc.class_.tally import Tally
import mcdc.mpi          as     mpi
from   mcdc.looper       import *
from   mcdc.print_       import print_banner, print_msg, print_runtime,\
                                print_progress_eigenvalue

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

def run():
    # Start timer
    mcdc.runtime_total.start()

    print_banner()
    prepare()
    
    print_msg("\n Now running TNT...")
    
    # SIMULATION LOOP
    simulation_end = False
    iter_idx = 0
    while not simulation_end:
        # Rebase rng skip_ahead seed
        mcdc.rng.skip_ahead(mcdc.mpi.work_start, rebase=True)

        # Loop over particel sources
        loop_source()
        
        # Tally closeout for eigenvalue mode
        if mcdc.settings.mode_eigenvalue:
            mcdc.tally.closeout(mcdc.settings.N_hist, iter_idx)

            # MPI reduce global tally
            mcdc.mpi.allreduce(mcdc.global_tally.nuSigmaF_sum, 
                               mcdc.global_tally.nuSigmaF_buff)
            if mcdc.settings.mode_alpha:
                mcdc.mpi.allreduce(mcdc.global_tally.ispeed_sum, 
                                   mcdc.global_tally.ispeed_buff)
            
            # Update k_eff
            mcdc.global_tally.k_eff = mcdc.global_tally.nuSigmaF_buff[0]\
                                 /mcdc.settings.N_hist
            if mcdc.mpi.master:
                mcdc.global_tally.k_mean[iter_idx] = mcdc.global_tally.k_eff
            
            # Update alpha_eff
            if mcdc.settings.mode_alpha:
                mcdc.global_tally.alpha_eff += \
                    (mcdc.global_tally.k_eff - 1.0)\
                    /(mcdc.global_tally.ispeed_buff[0]/mcdc.settings.N_hist)
                if mcdc.mpi.master:
                    mcdc.global_tally.alpha_mean[iter_idx] = \
                        mcdc.global_tally.alpha_eff
                        
            # Reset accumulators
            mcdc.global_tally.nuSigmaF_sum = 0.0
            if mcdc.settings.mode_alpha:
                mcdc.global_tally.ispeed_sum = 0.0
        
            # Progress printout
            print_progress_eigenvalue(iter_idx)

        # Simulation end?
        if mcdc.settings.mode_eigenvalue:
            iter_idx += 1
            if iter_idx == mcdc.settings.N_iter: simulation_end = True
        elif not mcdc.bank_stored: simulation_end = True

        # Manage particle banks
        if not simulation_end:
            if mcdc.settings.mode_eigenvalue:
                # Normalize weight
                mcdc.mpi.normalize_weight(mcdc.bank_stored, mcdc.settings.N_hist)

            # Rebase RNG for population control
            mcdc.rng.skip_ahead(
                mpi.work_size_total-mpi.work_start, rebase=True)

            # Population control
            mcdc.bank_stored = \
                mcdc.population_control(mcdc.bank_stored, mcdc.settings.N_hist)
            
            # Set stored bank as source bank for the next iteration
            mcdc.bank_source = mcdc.bank_stored
            mcdc.bank_stored = []
            if mcdc.settings.mode_eigenvalue:
                mcdc.bank_fission = mcdc.bank_stored
        else:
            mcdc.bank_source = []
            mcdc.bank_stored = []
            
    # Tally closeout for fixed-source mode
    if not mcdc.settings.mode_eigenvalue:
        mcdc.tally.closeout(mcdc.settings.N_hist, 0)

    generate_hdf5()
    
    # Stop timer
    mcdc.runtime_total.stop()
    
    print_runtime()

def prepare():
    print_msg(" Preparing...")

    # Tally
    if mcdc.tally is None:
        mcdc.tally = Tally('tally', ['flux'])
    mcdc.tally.allocate_bins()

    # To which bank fission neutrons are stored?
    if mcdc.settings.mode_eigenvalue:
        mcdc.bank_fission = mcdc.bank_stored
    else:
        mcdc.bank_fission = mcdc.bank_history

    # Population control
    mcdc.population_control.prepare(mcdc.settings.N_hist)

    # Distribute work to processors
    mcdc.mpi.distribute_work(mcdc.settings.N_hist)

def generate_hdf5():
    print_msg("\n\n Generating tally HDF5 files...\n")

    # Save tallies to HDF5
    if mcdc.mpi.master and mcdc.tally is not None:
        with h5py.File(mcdc.settings.output+'.h5', 'w') as f:
            # Runtime
            f.create_dataset("runtime",data=np.array([mcdc.runtime_total.total]))

            # Tally
            T = mcdc.tally
            f.create_dataset(T.name+"/grid/t", data=T.mesh.t)
            f.create_dataset(T.name+"/grid/x", data=T.mesh.x)
            f.create_dataset(T.name+"/grid/y", data=T.mesh.y)
            f.create_dataset(T.name+"/grid/z", data=T.mesh.z)
            
            # Scores
            for S in T.scores:
                f.create_dataset(T.name+"/"+S.name+"/mean",
                                 data=np.squeeze(S.mean))
                f.create_dataset(T.name+"/"+S.name+"/sdev",
                                 data=np.squeeze(S.sdev))
                S.mean.fill(0.0)
                S.sdev.fill(0.0)
                
            # Eigenvalues
            if mcdc.settings.mode_eigenvalue:
                f.create_dataset("keff",data=mcdc.global_tally.k_mean)
                mcdc.global_tally.k_mean.fill(0.0)
                if mcdc.settings.mode_alpha:
                    f.create_dataset("alpha_eff",data=mcdc.global_tally.alpha_mean)
                    mcdc.global_tally.alpha_mean.fill(0.0)
