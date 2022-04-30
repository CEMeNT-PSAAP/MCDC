import h5py
import numba as nb
import numpy as np

from   mcdc.class_.particle import type_particle
from   mcdc.class_.popctrl  import PCT_CO
from   mcdc.looper          import loop_source
import mcdc.mpi             as     mpi
from   mcdc.print_          import print_banner, print_msg, print_runtime,\
                                   print_progress_eigenvalue

# Get mcdc global variables as "mcdc"
import mcdc.global_ as mcdc_
mcdc = mcdc_.global_

def run():
    # Start timer
    mcdc.runtime_total = mpi.Wtime()

    # Print banner and configuration
    print_banner()
    if nb.config.DISABLE_JIT:
        print_msg("           Mode | Python")
    else:
        print_msg("           Mode | Numba")
    print_msg("  MPI Processes | %i"%mpi.size)
    print_msg(" OpenMP Threads | 1") # TODO

    # Preparation (memory allocation, etc)
    prepare()
    
    # Run
    print_msg(" Now running TNT...")
    
    # SIMULATION LOOP
    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)
        
        # Eigenvalue mode generation closeout
        if mcdc.setting.mode_eigenvalue:
            tally_closeout()
            tally_global_closeout()
            print_progress_eigenvalue()

        # Simulation end?
        if mcdc.setting.mode_eigenvalue:
            mcdc.i_iter += 1
            if mcdc.i_iter == mcdc.setting.N_iter: simulation_end = True
        elif not mcdc.bank.stored: simulation_end = True

        # Manage particle banks
        if not simulation_end:
            manage_particle_banks()
            
    # Fixed-source mode closeout
    if not mcdc.setting.mode_eigenvalue:
        tally_closeout()

    # Stop timer
    mcdc.runtime_total = mpi.Wtime() - mcdc.runtime_total
    
    # Generate output file
    generate_hdf5()
    
    print_runtime()

def prepare():
    print_msg("\n Preparing...")

    # Tally
    mcdc.tally.allocate_bins(mcdc.cells[0].material.G, mcdc.setting.N_iter)

    # To which bank fission neutrons are stored?
    if mcdc.setting.mode_eigenvalue:
        mcdc.bank.fission = mcdc.bank.stored

    # Population control
    if mcdc_.population_control is None:
        mcdc_.population_control = PCT_CO()
    mcdc_.population_control.prepare(mcdc.setting.N_hist)

    # Distribute work to processors
    mpi.distribute_work(mcdc.setting.N_hist)
    mcdc.N_work = mpi.work_size
    mcdc.master = mpi.master


def tally_closeout():
    if mcdc.tally.flux:
        score_closeout(mcdc.tally.score_flux)
    if mcdc.tally.current:
        score_closeout(mcdc.tally.score_current)
    if mcdc.tally.eddington:
        score_closeout(mcdc.tally.score_eddington)

def score_closeout(score):
    N_hist = mcdc.setting.N_hist
    i_iter = mcdc.i_iter

    # MPI Reduce
    score.sum_[:]   = mpi.reduce_master(score.sum_)
    score.sum_sq[:] = mpi.reduce_master(score.sum_sq)
    
    # Store results
    score.mean[i_iter,:] = score.sum_/N_hist
    score.sdev[i_iter,:] = np.sqrt((score.sum_sq/N_hist 
                                - np.square(score.mean[i_iter]))\
                               /(N_hist-1))
    
    # Reset history sums
    score.sum_.fill(0.0)
    score.sum_sq.fill(0.0)

def tally_global_closeout():
    N_hist = mcdc.setting.N_hist
    i_iter = mcdc.i_iter

    # MPI reduce
    mcdc.tally_global.nuSigmaF = mpi.allreduce(mcdc.tally_global.nuSigmaF)
    if mcdc.setting.mode_alpha:
        mcdc.tally_global.inverse_speed = \
            mpi.allreduce(mcdc.tally_global.inverse_speed)
    
    # Update and store k_eff
    mcdc.tally_global.k_eff = mcdc.tally_global.nuSigmaF/N_hist
    mcdc.tally_global.k_iterate[i_iter] = mcdc.tally_global.k_eff
    
    # Update and store alpha_eff
    if mcdc.setting.mode_alpha:
        k_eff         = mcdc.tally_global.k_eff
        inverse_speed = mcdc.tally_global.inverse_speed/N_hist

        mcdc.tally_global.alpha_eff += (k_eff - 1.0)/inverse_speed
        mcdc.tally_global.alpha_iterate[i_iter] = mcdc.tally_global.alpha_eff
                
    # Reset accumulators
    mcdc.tally_global.nuSigmaF = 0.0
    if mcdc.setting.mode_alpha:
        mcdc.tally_global.inverse_speed = 0.0        

def manage_particle_banks():
    if mcdc.setting.mode_eigenvalue:
        # Normalize weight
        mpi.normalize_weight(mcdc.bank.stored, mcdc.setting.N_hist)

    # Rebase RNG for population control
    mcdc.rng.skip_ahead_strides(mpi.work_size_total-mpi.work_start)
    mcdc.rng.rebase()

    # Population control
    mcdc.bank.stored = \
        mcdc_.population_control(mcdc.bank.stored, mcdc.setting.N_hist)
    mcdc.rng.rebase()
    
    # Update MPI-related global variables
    mcdc.N_work = mpi.work_size
    mcdc.master = mpi.master

    # Set stored bank as source bank for the next iteration
    mcdc.bank.source = mcdc.bank.stored
    mcdc.bank.stored = nb.typed.List.empty_list(type_particle)
    if mcdc.setting.mode_eigenvalue:
        mcdc.bank.fission = mcdc.bank.stored

def generate_hdf5():
    msg = " Generating tally HDF5 files..."
    if mcdc.setting.progress_bar: msg = "\n" + msg
    print_msg(msg)

    # Save tallies to HDF5
    if mpi.master:
        with h5py.File(mcdc.setting.output_name+'.h5', 'w') as f:
            # Runtime
            f.create_dataset("runtime",data=np.array([mcdc.runtime_total]))

            # Tally
            T = mcdc.tally
            f.create_dataset("tally/grid/t", data=T.mesh.t())
            f.create_dataset("tally/grid/x", data=T.mesh.x())
            f.create_dataset("tally/grid/y", data=T.mesh.y())
            f.create_dataset("tally/grid/z", data=T.mesh.z())
            
            # Scores
            if T.flux:
                f.create_dataset("tally/flux/mean",
                                 data=np.squeeze(T.score_flux.mean))
                f.create_dataset("tally/flux/sdev",
                                 data=np.squeeze(T.score_flux.sdev))
                T.score_flux.mean.fill(0.0)
                T.score_flux.sdev.fill(0.0)
            if T.current:
                f.create_dataset("tally/current/mean",
                                 data=np.squeeze(T.score_current.mean))
                f.create_dataset("tally/current/sdev",
                                 data=np.squeeze(T.score_current.sdev))
                T.score_current.mean.fill(0.0)
                T.score_current.sdev.fill(0.0)
            if T.eddington:
                f.create_dataset("tally/eddington/mean",
                                 data=np.squeeze(T.score_eddington.mean))
                f.create_dataset("tally/eddington/sdev",
                                 data=np.squeeze(T.score_eddington.sdev))
                T.score_eddington.mean.fill(0.0)
                T.score_eddington.sdev.fill(0.0)
                
            # Eigenvalues
            if mcdc.setting.mode_eigenvalue:
                f.create_dataset("keff",data=mcdc.tally_global.k_iterate)
                mcdc.tally_global.k_iterate.fill(0.0)
                if mcdc.setting.mode_alpha:
                    f.create_dataset("alpha_eff",data=mcdc.tally_global.alpha_iterate)
                    mcdc.tally_global.alpha_iterate.fill(0.0)
